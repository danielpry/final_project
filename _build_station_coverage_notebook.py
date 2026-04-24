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
        # NOAA Station Launch Coverage Investigation

        This notebook is rebuilt around the files in `data_final/NOAA_data`, which represent the NOAA/LCD files you were actually able to download.

        The notebook answers four questions:

        - Which launch facilities in the dataset now have a NOAA weather file in `data_final/NOAA_data`?
        - Which station ids are actually present in those downloaded files?
        - How much of each facility's launch history falls inside the downloaded weather date window?
        - After converting launch times into local standard time, how many launches can be matched to a weather observation within 2 hours?

        It also keeps a nearest-station reference table so you can compare the downloaded stations against the closest LCDv2 station inventory entries.
        """
    ),
    code(
        """
        from pathlib import Path
        import unicodedata

        import numpy as np
        import pandas as pd

        DATA_DIR = Path("data_final")
        NOAA_DIR = DATA_DIR / "NOAA_data"
        OUTPUT_DIR = DATA_DIR / "derived"
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

        STATION_LIST_PATH = NOAA_DIR / "lcdv2-station-list.txt"

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.max_colwidth", 200)
        """
    ),
    md("## Parse The LCDv2 Station Inventory"),
    code(
        """
        def load_lcdv2_station_list(path: Path) -> pd.DataFrame:
            rows = []
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.rstrip("\\n")
                    if not line.strip():
                        continue
                    rows.append(
                        {
                            "station_id": line[0:11].strip(),
                            "station_lat": pd.to_numeric(line[12:20].strip(), errors="coerce"),
                            "station_lon": pd.to_numeric(line[21:30].strip(), errors="coerce"),
                            "station_elevation_m": pd.to_numeric(line[31:37].strip(), errors="coerce"),
                            "station_name": line[41:].strip(),
                        }
                    )

            stations = pd.DataFrame(rows)
            stations = stations.dropna(subset=["station_lat", "station_lon"]).drop_duplicates("station_id").reset_index(drop=True)
            stations["station_country_hint"] = stations["station_id"].str[:2]
            return stations


        stations = load_lcdv2_station_list(STATION_LIST_PATH)
        print(f"Parsed {len(stations):,} stations from lcdv2-station-list.txt")
        stations.head()
        """
    ),
    md("## Rebuild Launch Facility Summary From `data_final`"),
    code(
        """
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


        launches = pd.read_csv(DATA_DIR / "Launches.csv")
        locations = pd.read_csv(DATA_DIR / "Locations.csv")

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
                first_launch_utc=("launch_time_utc", "min"),
                last_launch_utc=("launch_time_utc", "max"),
                first_launch_date=("launch_date", "min"),
                last_launch_date=("launch_date", "max"),
                raw_location_strings=("Location", "nunique"),
                launch_site_labels=("Launch Site", "nunique"),
                facility_lat=("facility_lat_candidate", "mean"),
                facility_lon=("facility_lon_candidate", "mean"),
            )
            .reset_index()
            .rename(columns={"Country_Code": "country_code", "Country": "country"})
            .sort_values("launches", ascending=False)
            .reset_index(drop=True)
        )

        coordinate_overrides = {
            "Pacific Spaceport Complex Alaska": {
                "facility_lat": 57.4350,
                "facility_lon": -152.3390,
                "coordinate_note": "manual override: Kodiak-area PSCA coordinate; Locations.csv appears inland",
            }
        }

        facility_summary["coordinate_note"] = "Locations.csv"
        for facility, override in coordinate_overrides.items():
            mask = facility_summary["facility"].eq(facility)
            facility_summary.loc[mask, "facility_lat"] = override["facility_lat"]
            facility_summary.loc[mask, "facility_lon"] = override["facility_lon"]
            facility_summary.loc[mask, "coordinate_note"] = override["coordinate_note"]

        print(f"{len(facility_summary):,} facilities have coordinates")
        print(f"{facility_summary['launches'].sum():,} launches map to those facilities")
        facility_summary.head(20)
        """
    ),
    md("## NOAA Files Actually Present In `data_final/NOAA_data`"),
    code(
        """
        FILE_TO_FACILITY = {
            "Alcantara_LC.csv": "Alcântara LC",
            "Baikonur_Cosmodrome.csv": "Baikonur Cosmodrome",
            "Base_Aerea_de_Gando.csv": "Base Aerea de Gando",
            "cape_canaveral_sfs.csv": "Cape Canaveral Space Force Station",
            "china_lake.csv": "China Lake",
            "edwards_afb.csv": "Edwards Air Force Base",
            "Guiana_SC.csv": "Guiana SC",
            "Imam_Khomeini_Spaceport.csv": "Imam Khomeini Spaceport",
            "Jiuquan_Satellite_LC.csv": "Jiuquan Satellite LC",
            "Kapustin_Yar.csv": "Kapustin Yar",
            "kennedy_sc.csv": "Kennedy Space Center",
            "Kiritimati_LA.csv": "Kiritimati LA",
            "Mahia_Peninsula.csv": "Māhia Peninsula",
            "Mojave_Air_and__Space_Port.csv": "Mojave Air and Space Port",
            "Naro_Space_Center.csv": "Naro Space Center",
            "pacific_missile_range_facility.csv": "Pacific Missile Range Facility",
            "Pacific_Spaceport_Complex_Alaska.csv": "Pacific Spaceport Complex Alaska",
            "Palmachim_Airbase.csv": "Palmachim Airbase",
            "Plesetsk_Cosmodrome.csv": "Plesetsk Cosmodrome",
            "RAAF_Woomera_RC.csv": "RAAF Woomera RC",
            "Ronald_Reagan_BMDTS.csv": "Ronald Reagan BMDTS",
            "Satish_Dhawan_SC.csv": "Satish Dhawan SC",
            "Shahrud_MTS.csv": "Shahrud MTS",
            "Sohae_SLS.csv": "Sohae SLS",
            "Svobodny_Cosmodrome.csv": "Svobodny Cosmodrome",
            "Taiyuan_Satellite_LC.csv": "Taiyuan Satellite LC",
            "Tanegashima_SC.csv": "Tanegashima SC",
            "Tonghae_SLG.csv": "Tonghae SLG",
            "Uchinoura_SC.csv": "Uchinoura SC",
            "vandenberg_sfb.csv": "Vandenberg Space Force Base",
            "Vostochny_Cosmodrome.csv": "Vostochny Cosmodrome",
            "wallops_flight_facility.csv": "Wallops Flight Facility",
            "Wenchang_Satellite_LC.csv": "Wenchang Satellite LC",
            "Xichang_Satellite_LC.csv": "Xichang Satellite LC",
            "Yasny_Cosmodrome.csv": "Yasny Cosmodrome",
        }

        FACILITY_UTC_OFFSET_HOURS = {
            "Alcântara LC": -3.0,
            "Baikonur Cosmodrome": 5.0,
            "Cape Canaveral Space Force Station": -5.0,
            "China Lake": -8.0,
            "Edwards Air Force Base": -8.0,
            "Guiana SC": -3.0,
            "Imam Khomeini Spaceport": 3.5,
            "Jiuquan Satellite LC": 8.0,
            "Kapustin Yar": 4.0,
            "Kennedy Space Center": -5.0,
            "Kiritimati LA": 14.0,
            "Māhia Peninsula": 12.0,
            "Mojave Air and Space Port": -8.0,
            "Naro Space Center": 9.0,
            "Pacific Missile Range Facility": -10.0,
            "Pacific Spaceport Complex Alaska": -9.0,
            "Palmachim Airbase": 2.0,
            "Plesetsk Cosmodrome": 3.0,
            "RAAF Woomera RC": 9.5,
            "Ronald Reagan BMDTS": 12.0,
            "Satish Dhawan SC": 5.5,
            "Shahrud MTS": 3.5,
            "Sohae SLS": 9.0,
            "Svobodny Cosmodrome": 9.0,
            "Taiyuan Satellite LC": 8.0,
            "Tanegashima SC": 9.0,
            "Tonghae SLG": 9.0,
            "Uchinoura SC": 9.0,
            "Vandenberg Space Force Base": -8.0,
            "Vostochny Cosmodrome": 9.0,
            "Wallops Flight Facility": -5.0,
            "Wenchang Satellite LC": 8.0,
            "Xichang Satellite LC": 8.0,
            "Yasny Cosmodrome": 5.0,
            "Base Aerea de Gando": 0.0,
        }


        def summarize_noaa_file(path: Path) -> dict:
            df = pd.read_csv(path, usecols=lambda c: c in {"STATION", "DATE"}, low_memory=False)
            dates = pd.to_datetime(df["DATE"], errors="coerce")
            station_ids = sorted(df["STATION"].dropna().astype(str).unique())
            return {
                "file_name": path.name,
                "facility": FILE_TO_FACILITY.get(path.name, path.stem),
                "file_path": str(path),
                "rows": len(df),
                "station_count": len(station_ids),
                "station_ids": ", ".join(station_ids),
                "weather_start_lstd": dates.min(),
                "weather_end_lstd": dates.max(),
            }


        noaa_file_inventory = pd.DataFrame(
            [summarize_noaa_file(path) for path in sorted(NOAA_DIR.glob("*.csv"))]
        ).sort_values(["facility", "file_name"]).reset_index(drop=True)

        noaa_file_inventory
        """
    ),
    md("## Downloaded Stations In Those Files"),
    code(
        """
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


        downloaded_station_rows = []
        for _, row in noaa_file_inventory.iterrows():
            facility = row["facility"]
            facility_row = facility_summary.loc[facility_summary["facility"] == facility]
            if facility_row.empty:
                continue
            facility_row = facility_row.iloc[0]

            station_ids = [s.strip() for s in str(row["station_ids"]).split(",") if s.strip()]
            for station_id in station_ids:
                station_row = stations.loc[stations["station_id"] == station_id]
                if station_row.empty:
                    downloaded_station_rows.append(
                        {
                            "facility": facility,
                            "file_name": row["file_name"],
                            "station_id": station_id,
                            "station_found_in_inventory": False,
                        }
                    )
                    continue

                station_row = station_row.iloc[0]
                distance = haversine_miles(
                    facility_row["facility_lat"],
                    facility_row["facility_lon"],
                    station_row["station_lat"],
                    station_row["station_lon"],
                )
                all_distances = haversine_miles(
                    facility_row["facility_lat"],
                    facility_row["facility_lon"],
                    stations["station_lat"].to_numpy(),
                    stations["station_lon"].to_numpy(),
                )
                rank = int((all_distances < distance).sum() + 1)
                downloaded_station_rows.append(
                    {
                        "facility": facility,
                        "country_code": facility_row["country_code"],
                        "launches": facility_row["launches"],
                        "file_name": row["file_name"],
                        "station_id": station_id,
                        "station_name": station_row["station_name"],
                        "station_lat": station_row["station_lat"],
                        "station_lon": station_row["station_lon"],
                        "station_elevation_m": station_row["station_elevation_m"],
                        "distance_miles": distance,
                        "distance_rank_among_all_lcdv2_stations": rank,
                        "station_found_in_inventory": True,
                    }
                )

        downloaded_station_summary = pd.DataFrame(downloaded_station_rows).sort_values(
            ["launches", "facility", "distance_miles"], ascending=[False, True, True]
        ).reset_index(drop=True)

        downloaded_station_summary
        """
    ),
    md("## Distance Between Launch Facilities And Downloaded Weather Stations"),
    code(
        """
        downloaded_station_distance_summary = (
            downloaded_station_summary.groupby(["facility", "country_code", "launches"], dropna=False)
            .agg(
                downloaded_station_count=("station_id", "nunique"),
                nearest_downloaded_station_id=("station_id", "first"),
                nearest_downloaded_station_name=("station_name", "first"),
                nearest_downloaded_station_distance_miles=("distance_miles", "min"),
                farthest_downloaded_station_distance_miles=("distance_miles", "max"),
                best_distance_rank_among_all_lcdv2_stations=("distance_rank_among_all_lcdv2_stations", "min"),
            )
            .reset_index()
            .sort_values(["launches", "nearest_downloaded_station_distance_miles"], ascending=[False, True])
            .reset_index(drop=True)
        )

        downloaded_station_distance_summary
        """
    ),
    md("## Three Closest LCDv2 Stations For Every Launch Facility"),
    code(
        """
        downloaded_station_pairs = {
            (row["facility"], row["station_id"])
            for _, row in downloaded_station_summary.dropna(subset=["station_id"]).iterrows()
        }

        nearest_rows = []
        for _, row in facility_summary.iterrows():
            distances = haversine_miles(
                row["facility_lat"],
                row["facility_lon"],
                stations["station_lat"].to_numpy(),
                stations["station_lon"].to_numpy(),
            )
            nearest = stations.copy()
            nearest["distance_miles"] = distances
            nearest = nearest.nsmallest(3, "distance_miles").copy()
            nearest["facility"] = row["facility"]
            nearest["country_code"] = row["country_code"]
            nearest["launches"] = row["launches"]
            nearest["first_launch_date"] = row["first_launch_date"]
            nearest["last_launch_date"] = row["last_launch_date"]
            nearest["facility_lat"] = row["facility_lat"]
            nearest["facility_lon"] = row["facility_lon"]
            nearest["coordinate_note"] = row["coordinate_note"]
            nearest["distance_rank"] = np.arange(1, len(nearest) + 1)
            nearest_rows.append(nearest)

        three_closest_stations = pd.concat(nearest_rows, ignore_index=True)
        three_closest_stations["downloaded_in_data_final"] = [
            (facility, station_id) in downloaded_station_pairs
            for facility, station_id in zip(
                three_closest_stations["facility"],
                three_closest_stations["station_id"],
            )
        ]

        three_closest_stations = three_closest_stations[
            [
                "facility",
                "country_code",
                "launches",
                "first_launch_date",
                "last_launch_date",
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
                "downloaded_in_data_final",
            ]
        ].sort_values(["launches", "facility", "distance_rank"], ascending=[False, True, True]).reset_index(drop=True)

        three_closest_stations
        """
    ),
    md("## Launch Date-Window Coverage From The Downloaded NOAA Files"),
    code(
        """
        facility_coverage_rows = []
        noaa_inventory_by_facility = noaa_file_inventory.set_index("facility")

        for _, facility_row in facility_summary.iterrows():
            facility = facility_row["facility"]
            launches_for_facility = launch_df.loc[launch_df["facility"] == facility].copy()
            launches_for_facility = launches_for_facility.dropna(subset=["launch_time_utc"]).copy()

            offset = FACILITY_UTC_OFFSET_HOURS.get(facility)
            launches_for_facility["launch_time_lstd"] = (
                launches_for_facility["launch_time_utc"] + pd.to_timedelta(offset, unit="h")
            ).dt.tz_localize(None) if offset is not None else pd.NaT

            if facility in noaa_inventory_by_facility.index:
                inv = noaa_inventory_by_facility.loc[facility]
                weather_start = inv["weather_start_lstd"]
                weather_end = inv["weather_end_lstd"]
                in_window = launches_for_facility["launch_time_lstd"].between(weather_start, weather_end, inclusive="both")
                before_window = launches_for_facility["launch_time_lstd"] < weather_start
                after_window = launches_for_facility["launch_time_lstd"] > weather_end
                file_exists = True
                station_ids = inv["station_ids"]
                file_name = inv["file_name"]
            else:
                weather_start = pd.NaT
                weather_end = pd.NaT
                in_window = pd.Series(False, index=launches_for_facility.index)
                before_window = pd.Series(False, index=launches_for_facility.index)
                after_window = pd.Series(False, index=launches_for_facility.index)
                file_exists = False
                station_ids = ""
                file_name = ""

            facility_coverage_rows.append(
                {
                    "facility": facility,
                    "country_code": facility_row["country_code"],
                    "launches": facility_row["launches"],
                    "first_launch_date": facility_row["first_launch_date"],
                    "last_launch_date": facility_row["last_launch_date"],
                    "file_exists": file_exists,
                    "file_name": file_name,
                    "station_ids": station_ids,
                    "utc_offset_hours": offset,
                    "weather_start_lstd": weather_start,
                    "weather_end_lstd": weather_end,
                    "launches_in_weather_window": int(in_window.sum()),
                    "launches_before_weather_start": int(before_window.sum()),
                    "launches_after_weather_end": int(after_window.sum()),
                    "date_window_coverage_rate": float(in_window.mean()) if len(launches_for_facility) else np.nan,
                }
            )

        facility_date_coverage = pd.DataFrame(facility_coverage_rows).sort_values(
            ["launches", "file_exists", "launches_in_weather_window"], ascending=[False, False, False]
        ).reset_index(drop=True)

        facility_date_coverage
        """
    ),
    md("## Nearest-Observation Weather Match Coverage Within 2 Hours"),
    code(
        """
        WEATHER_NUMERIC_COLUMNS = [
            "HourlyAltimeterSetting",
            "HourlyDryBulbTemperature",
            "HourlyDewPointTemperature",
            "HourlyRelativeHumidity",
            "HourlyPrecipitation",
            "HourlyVisibility",
            "HourlyStationPressure",
            "HourlySeaLevelPressure",
            "HourlyWindSpeed",
            "HourlyWindDirection",
            "HourlyWindGustSpeed",
            "HourlyWetBulbTemperature",
        ]

        WEATHER_TEXT_COLUMNS = [
            "HourlyPresentWeatherType",
            "HourlySkyConditions",
        ]

        SHORT_DURATION_PRECIP_COLUMNS = [
            "HourlyPrecipitation",
            "HourlyPrecipitation01Hour",
            "HourlyPrecipitation03Hour",
            "HourlyPrecipitation06Hour",
        ]


        def clean_lcd_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(
                series.astype(str).str.extract(r"([-+]?[0-9]*\\.?[0-9]+)")[0],
                errors="coerce",
            )


        def load_best_hourly_weather(path: Path) -> pd.DataFrame:
            weather_raw = pd.read_csv(path, low_memory=False)
            keep_cols = ["STATION", "DATE", "REPORT_TYPE"] + [
                c for c in WEATHER_NUMERIC_COLUMNS + WEATHER_TEXT_COLUMNS + SHORT_DURATION_PRECIP_COLUMNS
                if c in weather_raw.columns
            ]
            keep_cols = list(dict.fromkeys(keep_cols))
            weather = weather_raw[keep_cols].copy()
            weather = weather.loc[:, ~weather.columns.duplicated()].copy()

            for col in [c for c in WEATHER_NUMERIC_COLUMNS + SHORT_DURATION_PRECIP_COLUMNS if c in weather.columns]:
                weather[col] = clean_lcd_numeric(weather[col])

            weather["weather_obs_time_lstd"] = pd.to_datetime(weather["DATE"], errors="coerce")
            weather = weather.dropna(subset=["weather_obs_time_lstd"]).copy()
            weather["weather_obs_time_lstd"] = weather["weather_obs_time_lstd"].astype("datetime64[ns]")

            numeric_cols = [c for c in WEATHER_NUMERIC_COLUMNS if c in weather.columns]
            text_cols = [c for c in WEATHER_TEXT_COLUMNS if c in weather.columns]

            weather["hourly_nonnulls"] = weather[numeric_cols].notna().sum(axis=1)
            for col in text_cols:
                weather["hourly_nonnulls"] += weather[col].notna().astype(int)

            weather = (
                weather.loc[weather["hourly_nonnulls"] > 0]
                .sort_values(["weather_obs_time_lstd", "hourly_nonnulls"], ascending=[True, False])
                .drop_duplicates(subset=["weather_obs_time_lstd"], keep="first")
                .sort_values("weather_obs_time_lstd")
                .reset_index(drop=True)
            )
            return weather


        weather_match_rows = []
        matched_launch_frames = []

        for _, inv in noaa_file_inventory.sort_values("facility").iterrows():
            facility = inv["facility"]
            offset = FACILITY_UTC_OFFSET_HOURS.get(facility)
            if offset is None:
                continue

            facility_launches = launch_df.loc[launch_df["facility"] == facility].copy()
            facility_launches = facility_launches.dropna(subset=["launch_time_utc"]).copy()
            if facility_launches.empty:
                continue

            facility_launches["launch_time_lstd"] = (
                facility_launches["launch_time_utc"] + pd.to_timedelta(offset, unit="h")
            ).dt.tz_localize(None)
            facility_launches["launch_time_lstd"] = facility_launches["launch_time_lstd"].astype("datetime64[ns]")
            facility_launches = facility_launches.sort_values("launch_time_lstd")

            weather = load_best_hourly_weather(Path(inv["file_path"]))
            if weather.empty:
                weather_match_rows.append(
                    {
                        "facility": facility,
                        "launches": len(facility_launches),
                        "matched_launches": 0,
                        "match_rate": 0.0,
                        "median_abs_diff_minutes": np.nan,
                        "weather_rows_after_cleaning": 0,
                        "weather_start_lstd": pd.NaT,
                        "weather_end_lstd": pd.NaT,
                        "station_ids": inv["station_ids"],
                        "file_name": inv["file_name"],
                    }
                )
                continue

            merged = pd.merge_asof(
                facility_launches,
                weather,
                left_on="launch_time_lstd",
                right_on="weather_obs_time_lstd",
                direction="nearest",
                tolerance=pd.Timedelta("2h"),
            )

            merged["facility"] = facility
            merged["weather_matched"] = merged["weather_obs_time_lstd"].notna()
            merged["weather_time_diff_minutes"] = (
                (merged["launch_time_lstd"] - merged["weather_obs_time_lstd"]).dt.total_seconds().abs() / 60
            )
            merged["file_name"] = inv["file_name"]
            merged["station_ids"] = inv["station_ids"]
            matched_launch_frames.append(merged)

            weather_match_rows.append(
                {
                    "facility": facility,
                    "launches": len(facility_launches),
                    "matched_launches": int(merged["weather_matched"].sum()),
                    "match_rate": float(merged["weather_matched"].mean()),
                    "median_abs_diff_minutes": merged.loc[merged["weather_matched"], "weather_time_diff_minutes"].median(),
                    "weather_rows_after_cleaning": len(weather),
                    "weather_start_lstd": weather["weather_obs_time_lstd"].min(),
                    "weather_end_lstd": weather["weather_obs_time_lstd"].max(),
                    "station_ids": inv["station_ids"],
                    "file_name": inv["file_name"],
                }
            )

        weather_match_coverage = pd.DataFrame(weather_match_rows).sort_values(
            ["matched_launches", "match_rate"], ascending=[False, False]
        ).reset_index(drop=True)

        matched_launch_weather = pd.concat(matched_launch_frames, ignore_index=True) if matched_launch_frames else pd.DataFrame()

        weather_match_coverage
        """
    ),
    md("## Join Quality Focus"),
    code(
        """
        join_quality_summary = (
            facility_date_coverage.merge(
                weather_match_coverage[
                    [
                        "facility",
                        "matched_launches",
                        "match_rate",
                        "median_abs_diff_minutes",
                        "weather_rows_after_cleaning",
                    ]
                ],
                on="facility",
                how="left",
            )
            .merge(
                downloaded_station_distance_summary[
                    [
                        "facility",
                        "downloaded_station_count",
                        "nearest_downloaded_station_id",
                        "nearest_downloaded_station_name",
                        "nearest_downloaded_station_distance_miles",
                        "best_distance_rank_among_all_lcdv2_stations",
                    ]
                ],
                on="facility",
                how="left",
            )
        )

        join_quality_summary["window_to_match_conversion_rate"] = (
            join_quality_summary["matched_launches"] / join_quality_summary["launches_in_weather_window"]
        )
        join_quality_summary["unmatched_launches_inside_window"] = (
            join_quality_summary["launches_in_weather_window"] - join_quality_summary["matched_launches"]
        )
        join_quality_summary["outside_weather_window_launches"] = (
            join_quality_summary["launches"] - join_quality_summary["launches_in_weather_window"]
        )

        join_quality_summary = join_quality_summary.sort_values(
            ["launches", "match_rate", "date_window_coverage_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        join_quality_summary[
            [
                "facility",
                "country_code",
                "launches",
                "station_ids",
                "nearest_downloaded_station_id",
                "nearest_downloaded_station_distance_miles",
                "launches_in_weather_window",
                "date_window_coverage_rate",
                "matched_launches",
                "match_rate",
                "window_to_match_conversion_rate",
                "unmatched_launches_inside_window",
                "outside_weather_window_launches",
                "median_abs_diff_minutes",
            ]
        ]
        """
    ),
    md("## Facility Coverage Overview"),
    code(
        """
        facility_coverage_overview = (
            facility_summary.merge(
                facility_date_coverage[
                    [
                        "facility",
                        "file_exists",
                        "file_name",
                        "station_ids",
                        "utc_offset_hours",
                        "weather_start_lstd",
                        "weather_end_lstd",
                        "launches_in_weather_window",
                        "launches_before_weather_start",
                        "launches_after_weather_end",
                        "date_window_coverage_rate",
                    ]
                ],
                on="facility",
                how="left",
            )
            .merge(
                weather_match_coverage[
                    [
                        "facility",
                        "matched_launches",
                        "match_rate",
                        "median_abs_diff_minutes",
                        "weather_rows_after_cleaning",
                    ]
                ],
                on="facility",
                how="left",
            )
            .merge(
                downloaded_station_distance_summary[
                    [
                        "facility",
                        "downloaded_station_count",
                        "nearest_downloaded_station_id",
                        "nearest_downloaded_station_name",
                        "nearest_downloaded_station_distance_miles",
                        "farthest_downloaded_station_distance_miles",
                        "best_distance_rank_among_all_lcdv2_stations",
                    ]
                ],
                on="facility",
                how="left",
            )
            .sort_values(["launches", "file_exists", "matched_launches"], ascending=[False, False, False])
            .reset_index(drop=True)
        )

        facility_coverage_overview
        """
    ),
    md("## Count Of Launches With Weather Data Coverage"),
    code(
        """
        launch_weather_coverage_counts = pd.DataFrame(
            [
                {
                    "coverage_definition": "total launches",
                    "launch_count": int(facility_coverage_overview["launches"].sum()),
                },
                {
                    "coverage_definition": "launches at facilities with a downloaded NOAA file",
                    "launch_count": int(
                        facility_coverage_overview.loc[
                            facility_coverage_overview["file_exists"].fillna(False),
                            "launches",
                        ].sum()
                    ),
                },
                {
                    "coverage_definition": "launches inside downloaded weather date windows",
                    "launch_count": int(
                        facility_coverage_overview["launches_in_weather_window"].fillna(0).sum()
                    ),
                },
                {
                    "coverage_definition": "launches matched to weather within 2 hours",
                    "launch_count": int(
                        facility_coverage_overview["matched_launches"].fillna(0).sum()
                    ),
                },
            ]
        )

        launch_weather_coverage_counts["share_of_total_launches"] = (
            launch_weather_coverage_counts["launch_count"]
            / launch_weather_coverage_counts.loc[
                launch_weather_coverage_counts["coverage_definition"] == "total launches",
                "launch_count",
            ].iloc[0]
        )

        launch_weather_coverage_counts
        """
    ),
    md("## Remaining Gaps"),
    code(
        """
        missing_or_partial = facility_coverage_overview[
            [
                "facility",
                "country_code",
                "launches",
                "file_exists",
                "station_ids",
                "first_launch_date",
                "last_launch_date",
                "weather_start_lstd",
                "weather_end_lstd",
                "launches_in_weather_window",
                "matched_launches",
                "match_rate",
            ]
        ].copy()

        missing_or_partial["fully_covered_by_date_window"] = (
            missing_or_partial["launches_in_weather_window"] == missing_or_partial["launches"]
        )
        missing_or_partial["fully_matched_within_2h"] = (
            missing_or_partial["matched_launches"] == missing_or_partial["launches"]
        )

        missing_or_partial.loc[
            (~missing_or_partial["file_exists"])
            | (~missing_or_partial["fully_covered_by_date_window"])
            | (~missing_or_partial["fully_matched_within_2h"])
        ].sort_values(["launches", "file_exists", "matched_launches"], ascending=[False, True, True])
        """
    ),
    md("## Missing Facilities With LCDv2 Stations Within 50 Miles"),
    code(
        """
        missing_facilities = facility_coverage_overview.loc[
            ~facility_coverage_overview["file_exists"].fillna(False)
        ].copy()

        missing_station_radius_rows = []
        for _, row in missing_facilities.iterrows():
            distances = haversine_miles(
                row["facility_lat"],
                row["facility_lon"],
                stations["station_lat"].to_numpy(),
                stations["station_lon"].to_numpy(),
            )
            nearby = stations.copy()
            nearby["distance_miles"] = distances
            nearby = nearby.loc[nearby["distance_miles"] <= 50].sort_values("distance_miles").copy()

            if nearby.empty:
                missing_station_radius_rows.append(
                    {
                        "facility": row["facility"],
                        "country_code": row["country_code"],
                        "launches": row["launches"],
                        "first_launch_date": row["first_launch_date"],
                        "last_launch_date": row["last_launch_date"],
                        "facility_lat": row["facility_lat"],
                        "facility_lon": row["facility_lon"],
                        "stations_within_50_miles": 0,
                        "nearest_station_id": pd.NA,
                        "nearest_station_name": pd.NA,
                        "nearest_station_lat": pd.NA,
                        "nearest_station_lon": pd.NA,
                        "nearest_distance_miles": pd.NA,
                        "station_ids_within_50_miles": "",
                    }
                )
                continue

            nearest = nearby.iloc[0]
            missing_station_radius_rows.append(
                {
                    "facility": row["facility"],
                    "country_code": row["country_code"],
                    "launches": row["launches"],
                    "first_launch_date": row["first_launch_date"],
                    "last_launch_date": row["last_launch_date"],
                    "facility_lat": row["facility_lat"],
                    "facility_lon": row["facility_lon"],
                    "stations_within_50_miles": int(len(nearby)),
                    "nearest_station_id": nearest["station_id"],
                    "nearest_station_name": nearest["station_name"],
                    "nearest_station_lat": nearest["station_lat"],
                    "nearest_station_lon": nearest["station_lon"],
                    "nearest_distance_miles": nearest["distance_miles"],
                    "station_ids_within_50_miles": " | ".join(nearby["station_id"].tolist()),
                }
            )

        missing_facilities_within_50_miles = pd.DataFrame(missing_station_radius_rows).sort_values(
            ["launches", "stations_within_50_miles", "nearest_distance_miles"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        missing_facilities_within_50_miles
        """
    ),
    md("## Missing Facilities That Do Have At Least One Station Within 50 Miles"),
    code(
        """
        missing_facilities_within_50_miles.loc[
            missing_facilities_within_50_miles["stations_within_50_miles"] > 0
        ].reset_index(drop=True)
        """
    ),
    md("## Export Updated Coverage Tables"),
    code(
        """
        noaa_file_inventory.to_csv(OUTPUT_DIR / "noaa_file_inventory.csv", index=False)
        downloaded_station_summary.to_csv(OUTPUT_DIR / "downloaded_station_summary.csv", index=False)
        downloaded_station_distance_summary.to_csv(OUTPUT_DIR / "downloaded_station_distance_summary.csv", index=False)
        three_closest_stations.to_csv(OUTPUT_DIR / "lcdv2_three_closest_stations_by_launch_facility.csv", index=False)
        facility_date_coverage.to_csv(OUTPUT_DIR / "noaa_facility_date_window_coverage.csv", index=False)
        weather_match_coverage.to_csv(OUTPUT_DIR / "noaa_weather_match_coverage.csv", index=False)
        join_quality_summary.to_csv(OUTPUT_DIR / "noaa_join_quality_summary.csv", index=False)
        facility_coverage_overview.to_csv(OUTPUT_DIR / "noaa_facility_coverage_overview.csv", index=False)
        launch_weather_coverage_counts.to_csv(OUTPUT_DIR / "launch_weather_coverage_counts.csv", index=False)
        missing_facilities_within_50_miles.to_csv(
            OUTPUT_DIR / "missing_facilities_with_lcdv2_stations_within_50_miles.csv",
            index=False,
        )

        pd.DataFrame(
            {
                "file": [
                    str(OUTPUT_DIR / "noaa_file_inventory.csv"),
                    str(OUTPUT_DIR / "downloaded_station_summary.csv"),
                    str(OUTPUT_DIR / "downloaded_station_distance_summary.csv"),
                    str(OUTPUT_DIR / "lcdv2_three_closest_stations_by_launch_facility.csv"),
                    str(OUTPUT_DIR / "noaa_facility_date_window_coverage.csv"),
                    str(OUTPUT_DIR / "noaa_weather_match_coverage.csv"),
                    str(OUTPUT_DIR / "noaa_join_quality_summary.csv"),
                    str(OUTPUT_DIR / "noaa_facility_coverage_overview.csv"),
                    str(OUTPUT_DIR / "launch_weather_coverage_counts.csv"),
                    str(OUTPUT_DIR / "missing_facilities_with_lcdv2_stations_within_50_miles.csv"),
                ]
            }
        )
        """
    ),
    md("## Quick Totals"),
    code(
        """
        summary = pd.Series(
            {
                "facilities_in_launch_data": int(len(facility_summary)),
                "facilities_with_downloaded_noaa_file": int(facility_coverage_overview["file_exists"].fillna(False).sum()),
                "launches_total": int(facility_coverage_overview["launches"].sum()),
                "launches_with_downloaded_file": int(
                    facility_coverage_overview.loc[facility_coverage_overview["file_exists"].fillna(False), "launches"].sum()
                ),
                "launches_inside_downloaded_weather_window": int(
                    facility_coverage_overview["launches_in_weather_window"].fillna(0).sum()
                ),
                "launches_matched_within_2h": int(
                    facility_coverage_overview["matched_launches"].fillna(0).sum()
                ),
            }
        )

        summary
        """
    ),
]


nb["cells"] = cells
nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
