from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "noaa_station_launch_coverage_investigation.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells = [
    md(
        """
        # NOAA LCDv2 Station Coverage Investigation

        This notebook checks whether stations from `lcdv2-station-list.txt` are good candidates for the launch sites in `Launches.csv`/`Locations.csv`.

        Goals:

        - parse the LCDv2 station list into station id, latitude, longitude, elevation, and station name
        - rebuild the launch-site summary from the raw launch data
        - compare every launch facility to its nearest LCDv2 stations
        - identify whether the stations already downloaded in `data/*/*.csv` are the nearest practical choices
        - flag additional facilities and stations that could expand weather coverage beyond the current joins

        Important limitation: `lcdv2-station-list.txt` is a station inventory, not a per-station date coverage inventory. Distance is a necessary screen, but final NOAA pulls still need to confirm that hourly LCD data exists for the launch years.
        """
    ),
    code(
        """
        from pathlib import Path
        import math
        import re

        import numpy as np
        import pandas as pd

        DATA_DIR = Path("data")
        DERIVED_DIR = DATA_DIR / "derived"
        STATION_LIST_PATH = Path("lcdv2-station-list.txt")

        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_colwidth", 120)
        """
    ),
    md("## Load And Parse NOAA LCDv2 Stations"),
    code(
        r"""
        def load_lcdv2_station_list(path: Path) -> pd.DataFrame:
            rows = []
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    if not line.strip():
                        continue
                    # NOAA's station inventory is fixed-width in practice:
                    # station id, latitude, longitude, elevation, then station name.
                    station_id = line[0:11].strip()
                    lat = line[12:20].strip()
                    lon = line[21:30].strip()
                    elev = line[31:37].strip()
                    name = line[41:].strip()
                    rows.append(
                        {
                            "station_id": station_id,
                            "station_lat": pd.to_numeric(lat, errors="coerce"),
                            "station_lon": pd.to_numeric(lon, errors="coerce"),
                            "station_elevation_m": pd.to_numeric(elev, errors="coerce"),
                            "station_name": name,
                        }
                    )
            stations = pd.DataFrame(rows)
            stations = stations.dropna(subset=["station_lat", "station_lon"]).drop_duplicates("station_id")
            stations["station_country_prefix"] = stations["station_id"].str[:2]
            return stations

        stations = load_lcdv2_station_list(STATION_LIST_PATH)
        print(f"Parsed {len(stations):,} LCDv2 stations")
        stations.head()
        """
    ),
    md("## Rebuild Launch Facility Summary"),
    code(
        r"""
        launches = pd.read_csv(DATA_DIR / "Launches.csv")
        locations = pd.read_csv(DATA_DIR / "Locations.csv")

        def infer_facility_group(row) -> str:
            location = str(row.get("Location", ""))
            country_code = str(row.get("Country_Code", "")).upper()
            combined = row.get("Comb Launch Site")
            if country_code != "US":
                return combined

            text = location.lower()
            if "kennedy space center" in text:
                return "Kennedy Space Center"
            if "cape canaveral" in text:
                return "Cape Canaveral Space Force Station"
            if "vandenberg" in text:
                return "Vandenberg Space Force Base"
            if "wallops" in text:
                return "Wallops Flight Facility"
            if "pacific spaceport" in text or "kodiak" in text:
                return "Pacific Spaceport Complex Alaska"
            if "china lake" in text:
                return "China Lake"
            if "edwards" in text:
                return "Edwards Air Force Base"
            if "mojave" in text:
                return "Mojave Air and Space Port"
            if "kauai" in text or "pacific missile range" in text:
                return "Pacific Missile Range Facility"
            return combined

        launch_df = launches.copy()
        launch_df["launch_time_utc"] = pd.to_datetime(launch_df["Launch Time"], utc=True, errors="coerce")
        launch_df["launch_date"] = launch_df["launch_time_utc"].dt.date
        launch_df = launch_df.merge(
            locations,
            left_on="Location",
            right_on="Orig_Addr",
            how="left",
            suffixes=("", "_location"),
        )
        launch_df["Country_Code"] = launch_df["Country_Code"].astype(str).str.upper()
        launch_df["facility"] = launch_df.apply(infer_facility_group, axis=1)
        launch_df["facility_lat_candidate"] = np.where(
            launch_df["Country_Code"].eq("US"),
            launch_df["Lat"],
            launch_df["Comb Launch Site Lat"],
        )
        launch_df["facility_lon_candidate"] = np.where(
            launch_df["Country_Code"].eq("US"),
            launch_df["Lon"],
            launch_df["Comb Launch Site Lon"],
        )

        facility_summary = (
            launch_df.dropna(subset=["facility", "facility_lat_candidate", "facility_lon_candidate"])
            .groupby(["facility", "Country", "Country_Code"], dropna=False)
            .agg(
                launches=("Launch Id", "count"),
                first_launch=("launch_date", "min"),
                last_launch=("launch_date", "max"),
                raw_location_strings=("Location", "nunique"),
                launch_site_labels=("Launch Site", "nunique"),
                facility_lat=("facility_lat_candidate", "mean"),
                facility_lon=("facility_lon_candidate", "mean"),
            )
            .reset_index()
            .rename(columns={"Country_Code": "country_code"})
            .sort_values("launches", ascending=False)
        )

        # Locations.csv places Pacific Spaceport Complex Alaska near interior Alaska
        # (64.200841, -149.493673), which is inconsistent with the Kodiak launch
        # site and with the already-downloaded Kodiak/Akhiok LCD stations.
        coordinate_overrides = {
            "Pacific Spaceport Complex Alaska": {
                "facility_lat": 57.4350,
                "facility_lon": -152.3390,
                "coordinate_note": "manual override: Kodiak-area PSCA coordinate; Locations.csv appears inland",
            }
        }
        facility_summary["source_facility_lat"] = facility_summary["facility_lat"]
        facility_summary["source_facility_lon"] = facility_summary["facility_lon"]
        facility_summary["coordinate_note"] = "Locations.csv"
        for facility, override in coordinate_overrides.items():
            mask = facility_summary["facility"] == facility
            for col in ["facility_lat", "facility_lon", "coordinate_note"]:
                facility_summary.loc[mask, col] = override[col]

        print(f"{len(facility_summary):,} launch facilities have coordinates")
        print(f"{facility_summary['launches'].sum():,} launches have facility coordinates")
        facility_summary.head(20)
        """
    ),
    md("## Current Weather Pulls And Stations Already Used"),
    code(
        r"""
        facility_file_map = {
            "Baikonur Cosmodrome": DATA_DIR / "baikonur_cosmodrome" / "Baikonur_Cosmodrome.csv",
            "Cape Canaveral Space Force Station": DATA_DIR / "cape_canaveral_sfs" / "cape_canaveral_sfs.csv",
            "China Lake": DATA_DIR / "china_lake" / "china_lake.csv",
            "Edwards Air Force Base": DATA_DIR / "edwards_afb" / "edwards_afb.csv",
            "Kennedy Space Center": DATA_DIR / "kennedy_sc" / "kennedy_sc.csv",
            "Mojave Air and Space Port": DATA_DIR / "mojave_air_space_port" / "mojave_air_space_port.csv",
            "Pacific Missile Range Facility": DATA_DIR / "pacific_missile_range" / "pacific_missile_range.csv",
            "Pacific Spaceport Complex Alaska": DATA_DIR / "pacific_spaceport_alaska" / "pacific_spaceport_alaska.csv",
            "Plesetsk Cosmodrome": DATA_DIR / "plesetsk_cosmodrome" / "Plesetsk_Cosmodrome.csv",
            "Vandenberg Space Force Base": DATA_DIR / "vandenberg_sfb" / "vandenberg_sfb.csv",
            "Wallops Flight Facility": DATA_DIR / "wallops_flight_facility" / "wallops_flight_facility.csv",
        }

        def summarize_weather_file(facility: str, path: Path) -> dict:
            if not path.exists():
                return {"facility": facility, "weather_file": str(path), "file_exists": False}
            df = pd.read_csv(path, usecols=lambda c: c in {"STATION", "DATE"}, low_memory=False)
            dates = pd.to_datetime(df["DATE"], errors="coerce")
            station_ids = sorted(df["STATION"].dropna().astype(str).unique())
            return {
                "facility": facility,
                "weather_file": str(path),
                "file_exists": True,
                "current_station_ids": ", ".join(station_ids),
                "current_station_count": len(station_ids),
                "weather_rows": len(df),
                "weather_start": dates.min(),
                "weather_end": dates.max(),
            }

        current_weather = pd.DataFrame(
            [summarize_weather_file(facility, path) for facility, path in facility_file_map.items()]
        )

        current_weather
        """
    ),
    md("## Distance Utility"),
    code(
        r"""
        def haversine_miles(lat1, lon1, lat2, lon2):
            radius_miles = 3958.7613
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            lat2 = np.radians(lat2)
            lon2 = np.radians(lon2)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            return 2 * radius_miles * np.arcsin(np.sqrt(a))

        def nearest_stations_for_facility(row, top_n=10):
            distances = haversine_miles(
                row["facility_lat"],
                row["facility_lon"],
                stations["station_lat"].to_numpy(),
                stations["station_lon"].to_numpy(),
            )
            nearest = stations.copy()
            nearest["distance_miles"] = distances
            nearest = nearest.nsmallest(top_n, "distance_miles").copy()
            nearest.insert(0, "facility", row["facility"])
            nearest.insert(1, "country_code", row["country_code"])
            nearest.insert(2, "launches", row["launches"])
            nearest.insert(3, "first_launch", row["first_launch"])
            nearest.insert(4, "last_launch", row["last_launch"])
            nearest.insert(5, "facility_lat", row["facility_lat"])
            nearest.insert(6, "facility_lon", row["facility_lon"])
            nearest.insert(7, "coordinate_note", row.get("coordinate_note", "Locations.csv"))
            nearest["distance_rank"] = np.arange(1, len(nearest) + 1)
            return nearest

        nearest_all = pd.concat(
            [nearest_stations_for_facility(row, top_n=10) for _, row in facility_summary.iterrows()],
            ignore_index=True,
        )

        nearest_all.head(20)
        """
    ),
    md("## Are Current Stations The Nearest Options?"),
    code(
        r"""
        current_station_rows = []
        current_weather_ids = current_weather.dropna(subset=["current_station_ids"]).copy()
        for _, current in current_weather_ids.iterrows():
            facility = current["facility"]
            facility_row = facility_summary.loc[facility_summary["facility"] == facility]
            if facility_row.empty:
                continue
            facility_row = facility_row.iloc[0]
            ids = [s.strip() for s in str(current["current_station_ids"]).split(",") if s.strip()]
            for station_id in ids:
                station = stations.loc[stations["station_id"] == station_id]
                if station.empty:
                    current_station_rows.append(
                        {
                            "facility": facility,
                            "current_station_id": station_id,
                            "current_station_found_in_lcdv2_list": False,
                        }
                    )
                    continue
                station = station.iloc[0]
                distance = haversine_miles(
                    facility_row["facility_lat"],
                    facility_row["facility_lon"],
                    station["station_lat"],
                    station["station_lon"],
                )
                rank = int(
                    (haversine_miles(
                        facility_row["facility_lat"],
                        facility_row["facility_lon"],
                        stations["station_lat"].to_numpy(),
                        stations["station_lon"].to_numpy(),
                    ) < distance).sum()
                    + 1
                )
                current_station_rows.append(
                    {
                        "facility": facility,
                        "launches": facility_row["launches"],
                        "country_code": facility_row["country_code"],
                        "current_station_id": station_id,
                        "current_station_name": station["station_name"],
                        "current_station_found_in_lcdv2_list": True,
                        "current_station_distance_miles": distance,
                        "current_station_distance_rank": rank,
                    }
                )

        current_station_comparison = pd.DataFrame(current_station_rows).merge(
            current_weather,
            on="facility",
            how="left",
        )

        current_station_comparison.sort_values(["current_station_distance_rank", "launches"], ascending=[False, False])
        """
    ),
    md("## Three Closest Stations For Every Launch Facility"),
    code(
        r"""
        current_pairs = set()
        for _, row in current_station_comparison.dropna(subset=["current_station_id"]).iterrows():
            current_pairs.add((row["facility"], row["current_station_id"]))

        three_closest_stations = nearest_all.loc[nearest_all["distance_rank"] <= 3].copy()
        three_closest_stations["is_current_downloaded_station"] = [
            (facility, station_id) in current_pairs
            for facility, station_id in zip(
                three_closest_stations["facility"],
                three_closest_stations["station_id"],
            )
        ]
        three_closest_stations["station_country_hint"] = three_closest_stations["station_id"].str[:2]
        three_closest_stations = three_closest_stations[
            [
                "facility",
                "country_code",
                "launches",
                "first_launch",
                "last_launch",
                "facility_lat",
                "facility_lon",
                "coordinate_note",
                "distance_rank",
                "station_id",
                "station_name",
                "station_country_hint",
                "station_lat",
                "station_lon",
                "station_elevation_m",
                "distance_miles",
                "is_current_downloaded_station",
            ]
        ].sort_values(["launches", "facility", "distance_rank"], ascending=[False, True, True])

        three_closest_stations
        """
    ),
    md("## Best Nearby Station Candidates By Facility"),
    code(
        r"""
        top_nearest = nearest_all.loc[nearest_all["distance_rank"] <= 5].copy()

        top_nearest["is_current_downloaded_station"] = [
            (facility, station_id) in current_pairs
            for facility, station_id in zip(top_nearest["facility"], top_nearest["station_id"])
        ]

        top_nearest[
            [
                "facility",
                "country_code",
                "launches",
                "first_launch",
                "last_launch",
                "distance_rank",
                "station_id",
                "station_name",
                "distance_miles",
                "coordinate_note",
                "station_lat",
                "station_lon",
                "is_current_downloaded_station",
            ]
        ].sort_values(["launches", "facility", "distance_rank"], ascending=[False, True, True]).head(75)
        """
    ),
    md("## Spatial Coverage Counts"),
    code(
        r"""
        nearest_1 = nearest_all.loc[nearest_all["distance_rank"] == 1].copy()
        thresholds = [10, 25, 50, 100, 250]
        coverage_rows = []
        for radius in thresholds:
            in_radius = nearest_1["distance_miles"] <= radius
            coverage_rows.append(
                {
                    "radius_miles": radius,
                    "facilities_with_station": int(in_radius.sum()),
                    "launches_with_station": int(nearest_1.loc[in_radius, "launches"].sum()),
                    "share_of_coordinate_launches": nearest_1.loc[in_radius, "launches"].sum() / nearest_1["launches"].sum(),
                }
            )
        spatial_coverage = pd.DataFrame(coverage_rows)
        spatial_coverage
        """
    ),
    md("## High-Value Facilities Not Yet Downloaded"),
    code(
        r"""
        downloaded_facilities = set(current_weather.loc[current_weather["file_exists"], "facility"])
        facility_opportunities = nearest_1.copy()
        facility_opportunities["currently_downloaded"] = facility_opportunities["facility"].isin(downloaded_facilities)
        facility_opportunities["nearest_station_within_50_miles"] = facility_opportunities["distance_miles"] <= 50
        facility_opportunities["nearest_station_within_100_miles"] = facility_opportunities["distance_miles"] <= 100

        opportunity_cols = [
            "facility",
            "country_code",
            "launches",
            "first_launch",
            "last_launch",
            "station_id",
            "station_name",
            "distance_miles",
            "station_lat",
            "station_lon",
            "currently_downloaded",
            "nearest_station_within_50_miles",
            "nearest_station_within_100_miles",
        ]

        facility_opportunities.loc[
            ~facility_opportunities["currently_downloaded"], opportunity_cols
        ].sort_values(["launches", "distance_miles"], ascending=[False, True]).head(40)
        """
    ),
    md("## Candidate Pull Reference"),
    code(
        r"""
        candidate_pull_reference = top_nearest[
            [
                "facility",
                "country_code",
                "launches",
                "first_launch",
                "last_launch",
                "distance_rank",
                "station_id",
                "station_name",
                "station_lat",
                "station_lon",
                "station_elevation_m",
                "distance_miles",
                "coordinate_note",
                "is_current_downloaded_station",
            ]
        ].copy()

        candidate_pull_reference["recommended_for_screening"] = (
            (candidate_pull_reference["distance_rank"] <= 3)
            | candidate_pull_reference["is_current_downloaded_station"]
            | (candidate_pull_reference["distance_miles"] <= 25)
        )

        out_path = DERIVED_DIR / "lcdv2_station_launch_candidate_reference.csv"
        candidate_pull_reference.to_csv(out_path, index=False)
        print(f"Wrote {out_path} with {len(candidate_pull_reference):,} rows")
        candidate_pull_reference.head(30)
        """
    ),
    md("## Save Three-Closest-Station Reference"),
    code(
        r"""
        three_closest_out_path = DERIVED_DIR / "lcdv2_three_closest_stations_by_launch_facility.csv"
        three_closest_stations.to_csv(three_closest_out_path, index=False)
        print(f"Wrote {three_closest_out_path} with {len(three_closest_stations):,} rows")
        three_closest_stations.head(30)
        """
    ),
    md("## Practical Findings To Check After Running"),
    code(
        r"""
        summary = {
            "launch_facilities_with_coordinates": len(facility_summary),
            "launches_with_facility_coordinates": int(facility_summary["launches"].sum()),
            "currently_downloaded_facilities": len(downloaded_facilities),
            "not_downloaded_facilities": int((~facility_opportunities["currently_downloaded"]).sum()),
            "not_downloaded_launches": int(facility_opportunities.loc[~facility_opportunities["currently_downloaded"], "launches"].sum()),
            "not_downloaded_launches_with_station_within_50_miles": int(
                facility_opportunities.loc[
                    (~facility_opportunities["currently_downloaded"])
                    & (facility_opportunities["nearest_station_within_50_miles"]),
                    "launches",
                ].sum()
            ),
        }
        pd.Series(summary)
        """
    ),
    md(
        """
        ## Interpretation Notes

        Use this notebook as a station-screening step, not as final proof of usable weather joins.

        Recommended follow-up checks for any high-value candidate station:

        - confirm NOAA LCD hourly data exists across the facility's launch date range
        - download the station CSV and run the same nearest-hour weather merge used in the existing project
        - compare match rate and weather-feature availability against the currently downloaded station
        - prefer nearby coastal/island stations only when their meteorological setting is plausibly representative of the launch site
        """
    ),
]

nb["cells"] = cells

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
