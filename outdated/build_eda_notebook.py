import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "EDA.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """
        # Rocket Launch EDA for NOAA Weather Join Planning

        This notebook profiles the launch data in `data/` with one practical goal:
        identify the date coverage and U.S. ZIP-code coverage needed to pull NOAA weather
        data for a launch-weather modeling project.

        The analysis focuses on:
        - launch date coverage
        - location normalization for U.S. launch sites
        - candidate ZIP codes for NOAA extraction
        - data quality issues that matter before weather joins
        """
    ),
    code(
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.max_colwidth", 120)
        sns.set_theme(style="whitegrid", palette="deep")

        DATA_DIR = Path("data")
        OUTPUT_DIR = DATA_DIR / "derived"
        OUTPUT_DIR.mkdir(exist_ok=True)

        WEATHER_NUMERIC_COLUMNS = [
            "HourlyAltimeterSetting",
            "HourlyDryBulbTemperature",
            "HourlyDewPointTemperature",
            "HourlyRelativeHumidity",
            "HourlyPrecipitation",
            "HourlyVisibility",
            "HourlyStationPressure",
            "HourlySeaLevelPressure",
            "HourlyWetBulbTemperature",
            "HourlyWindSpeed",
            "HourlyWindGustSpeed",
            "HourlyWindDirection",
        ]
        WEATHER_TEXT_COLUMNS = ["HourlyPresentWeatherType", "HourlySkyConditions"]
        SHORT_DURATION_PRECIP_COLUMNS = [
            "ShortDurationPrecipitationValue005",
            "ShortDurationPrecipitationValue010",
            "ShortDurationPrecipitationValue015",
            "ShortDurationPrecipitationValue020",
            "ShortDurationPrecipitationValue030",
            "ShortDurationPrecipitationValue045",
            "ShortDurationPrecipitationValue060",
            "ShortDurationPrecipitationValue080",
            "ShortDurationPrecipitationValue100",
            "ShortDurationPrecipitationValue120",
            "ShortDurationPrecipitationValue150",
            "ShortDurationPrecipitationValue180",
        ]
        WEATHER_FILE_MAP = {
            "Cape Canaveral Space Force Station": ("cape_canaveral_sfs/cape_canaveral_sfs.csv", -5),
            "Kennedy Space Center": ("kennedy_sc/kennedy_sc.csv", -5),
            "Vandenberg Space Force Base": ("vandenberg_sfb/vandenberg_sfb.csv", -8),
            "Wallops Flight Facility": ("wallops_flight_facility/wallops_flight_facility.csv", -5),
            "Pacific Spaceport Complex Alaska": ("pacific_spaceport_alaska/pacific_spaceport_alaska.csv", -9),
            "Pacific Missile Range Facility": ("pacific_missile_range/pacific_missile_range.csv", -10),
            "Mojave Air and Space Port": ("mojave_air_space_port/mojave_air_space_port.csv", -8),
            "Edwards Air Force Base": ("edwards_afb/edwards_afb.csv", -8),
            "China Lake": ("china_lake/china_lake.csv", -8),
        }
        """
    ),
    code(
        """
        companies = pd.read_csv(DATA_DIR / "Companies.csv")
        configs = pd.read_csv(DATA_DIR / "Configs.csv")
        families = pd.read_csv(DATA_DIR / "Families.csv")
        launches = pd.read_csv(DATA_DIR / "Launches.csv")
        locations = pd.read_csv(DATA_DIR / "Locations.csv")
        missions = pd.read_csv(DATA_DIR / "Missions.csv")

        dataset_shapes = pd.DataFrame(
            {
                "dataset": [
                    "Companies",
                    "Configs",
                    "Families",
                    "Launches",
                    "Locations",
                    "Missions",
                ],
                "rows": [
                    len(companies),
                    len(configs),
                    len(families),
                    len(launches),
                    len(locations),
                    len(missions),
                ],
                "columns": [
                    companies.shape[1],
                    configs.shape[1],
                    families.shape[1],
                    launches.shape[1],
                    locations.shape[1],
                    missions.shape[1],
                ],
            }
        )

        dataset_shapes
        """
    ),
    md(
        """
        ## Global Launch Site Footprint

        Before narrowing to the current U.S.-focused NOAA workflow, it is useful to understand
        the full launch-site footprint in the raw dataset. This section looks across **all**
        launches and all joined launch sites so you can evaluate how realistic it would be to
        expand the project to non-U.S. weather sources.

        The main questions are:
        - which countries account for the most launches
        - which combined launch sites dominate the data volume
        - whether the raw dataset already contains usable coordinates for those sites
        - which non-U.S. sites would be the highest-priority targets for future weather sourcing
        """
    ),
    code(
        """
        global_launch_scope = launches.merge(
            locations[
                [
                    "Orig_Addr",
                    "Country",
                    "Country_Code",
                    "Launch Site",
                    "Comb Launch Site",
                    "Lat",
                    "Lon",
                    "Comb Launch Site Lat",
                    "Comb Launch Site Lon",
                ]
            ],
            left_on="Location",
            right_on="Orig_Addr",
            how="left",
        ).copy()

        global_launch_scope["Country_Code"] = global_launch_scope["Country_Code"].astype(str).str.upper()
        global_launch_scope["country_joined"] = global_launch_scope["Country"].notna()
        global_launch_scope["combined_site_joined"] = global_launch_scope["Comb Launch Site"].notna()

        scope_join_summary = pd.DataFrame(
            {
                "metric": [
                    "total launches",
                    "unique raw location strings",
                    "launches with country joined",
                    "launches with combined site joined",
                    "unique countries in joined scope",
                    "unique combined launch sites in joined scope",
                ],
                "value": [
                    len(global_launch_scope),
                    global_launch_scope["Location"].nunique(),
                    int(global_launch_scope["country_joined"].sum()),
                    int(global_launch_scope["combined_site_joined"].sum()),
                    int(global_launch_scope["Country"].nunique()),
                    int(global_launch_scope["Comb Launch Site"].nunique()),
                ],
            }
        )

        scope_join_summary
        """
    ),
    md(
        """
        This quick audit shows whether the raw launch records can already be mapped to countries
        and combined sites. If most launches join cleanly, then extending the project beyond the
        U.S. is mainly a **weather-source acquisition** problem rather than a launch-site
        normalization problem.
        """
    ),
    code(
        """
        country_scope_summary = (
            global_launch_scope.groupby(["Country", "Country_Code"], dropna=False)
            .agg(
                launches=("Launch Id", "count"),
                unique_raw_locations=("Location", "nunique"),
                unique_combined_sites=("Comb Launch Site", "nunique"),
            )
            .reset_index()
            .sort_values(["launches", "unique_combined_sites"], ascending=[False, False])
        )

        country_scope_summary.head(20)
        """
    ),
    code(
        """
        plt.figure(figsize=(12, 6))
        top_countries = country_scope_summary.head(12).copy()
        sns.barplot(data=top_countries, y="Country", x="launches", color="#4e79a7")
        plt.title("Top countries in the full launch dataset")
        plt.xlabel("Launch count")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        This table and chart help define the practical expansion path. If the goal is to broaden
        weather coverage, the highest-leverage countries are the ones with both substantial launch
        counts and repeated site usage, because those are the locations where a weather-data
        investment would affect the most rows.
        """
    ),
    code(
        """
        site_scope_summary = (
            global_launch_scope.groupby(
                [
                    "Country",
                    "Country_Code",
                    "Comb Launch Site",
                    "Comb Launch Site Lat",
                    "Comb Launch Site Lon",
                ],
                dropna=False,
            )
            .agg(
                launches=("Launch Id", "count"),
                unique_raw_locations=("Location", "nunique"),
            )
            .reset_index()
            .sort_values(["launches", "Country", "Comb Launch Site"], ascending=[False, True, True])
        )

        site_scope_summary.head(25)
        """
    ),
    code(
        """
        mapped_sites = site_scope_summary.dropna(subset=["Comb Launch Site Lat", "Comb Launch Site Lon"]).copy()
        mapped_sites["site_label"] = mapped_sites["Comb Launch Site"] + " (" + mapped_sites["Country_Code"] + ")"

        plt.figure(figsize=(14, 7))
        sns.scatterplot(
            data=mapped_sites,
            x="Comb Launch Site Lon",
            y="Comb Launch Site Lat",
            size="launches",
            hue="Country_Code",
            sizes=(40, 700),
            alpha=0.8,
            legend="brief",
        )
        plt.title("Global launch-site footprint from the full dataset")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        The latitude/longitude plot is not a political map, but it is enough for planning
        purposes. It shows where the major launch clusters sit geographically and whether the
        dataset already has site-level coordinates that could support future weather joins from
        non-NOAA sources.
        """
    ),
    code(
        """
        non_us_site_scope = (
            site_scope_summary[site_scope_summary["Country_Code"] != "US"]
            .copy()
            .reset_index(drop=True)
        )

        non_us_site_scope.head(25)
        """
    ),
    code(
        """
        non_us_country_scope = (
            country_scope_summary[country_scope_summary["Country_Code"] != "US"]
            .copy()
            .reset_index(drop=True)
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=non_us_country_scope.head(12), y="Country", x="launches", color="#59a14f")
        plt.title("Top non-U.S. countries by launch count")
        plt.xlabel("Launch count")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        This non-U.S. summary is the most directly useful table for the project-expansion question.
        It identifies the countries and combined launch sites that would matter most if you decide
        to research international weather data sources. In practice, the likely first targets are
        the biggest repeated non-U.S. sites, such as major launch complexes in Russia,
        Kazakhstan, French Guiana, China, Japan, and India.
        """
    ),
    md(
        """
        ## Launch-Level Preparation

        `Launches.csv` is the core table for weather joins. We convert timestamps, join site
        metadata from `Locations.csv`, and derive a few fields that make the NOAA prep work
        easier.
        """
    ),
    code(
        """
        def parse_numeric_text(series: pd.Series) -> pd.Series:
            return pd.to_numeric(
                series.astype(str).str.extract(r"([-+]?[0-9]*\\.?[0-9]+)")[0],
                errors="coerce",
            )


        launch_df = launches.copy()
        launch_df["launch_time_utc"] = pd.to_datetime(launch_df["Launch Time"], utc=True, errors="coerce")
        launch_df["launch_date"] = launch_df["launch_time_utc"].dt.date
        launch_df["launch_year"] = launch_df["launch_time_utc"].dt.year
        launch_df["launch_month"] = launch_df["launch_time_utc"].dt.month
        launch_df["launch_month_name"] = launch_df["launch_time_utc"].dt.month_name()
        launch_df["launch_decade"] = (launch_df["launch_year"] // 10) * 10

        mission_agg = (
            missions.groupby("Launch Id")
            .agg(
                payload_count=("Payloads", "sum"),
                mission_mass=("Mass", "sum"),
                mission_rows=("No", "count"),
            )
            .reset_index()
        )

        config_features = configs.merge(
            families[["Family Id", "Family", "Success Rate"]],
            on="Family Id",
            how="left",
        ).copy()

        for col in [
            "Liftoff Thrust",
            "Payload to LEO",
            "Payload to GTO",
            "Stages",
            "Strap-ons",
            "Rocket Height",
            "Fairing Diameter",
            "Fairing Height",
        ]:
            if col in config_features.columns:
                config_features[col] = parse_numeric_text(config_features[col])

        config_features["family_success_rate_pct"] = (
            config_features["Success Rate"].astype(str).str.rstrip("%")
        )
        config_features["family_success_rate_pct"] = pd.to_numeric(
            config_features["family_success_rate_pct"],
            errors="coerce",
        )

        launch_df = launch_df.merge(
            locations[
                [
                    "Orig_Addr",
                    "Country",
                    "Country_Code",
                    "Operator",
                    "Launch Site",
                    "Comb Launch Site",
                    "Lat",
                    "Lon",
                    "Comb Launch Site Lat",
                    "Comb Launch Site Lon",
                ]
            ],
            left_on="Location",
            right_on="Orig_Addr",
            how="left",
        )
        launch_df = launch_df.merge(mission_agg, on="Launch Id", how="left")
        launch_df = launch_df.merge(
            config_features[
                [
                    "Config",
                    "Status",
                    "Liftoff Thrust",
                    "Payload to LEO",
                    "Payload to GTO",
                    "Stages",
                    "Strap-ons",
                    "Rocket Height",
                    "Fairing Diameter",
                    "Fairing Height",
                    "Family",
                    "family_success_rate_pct",
                ]
            ],
            left_on="Rocket Name",
            right_on="Config",
            how="left",
        )

        launch_df["location_joined"] = launch_df["Orig_Addr"].notna()
        launch_df = launch_df.rename(
            columns={
                "Payload to LEO": "config_payload_leo",
                "Payload to GTO": "config_payload_gto",
                "Status": "config_status",
                "Liftoff Thrust": "config_liftoff_thrust",
                "Stages": "config_stages",
                "Strap-ons": "config_strap_ons",
                "Rocket Height": "config_rocket_height",
                "Fairing Diameter": "config_fairing_diameter",
                "Fairing Height": "config_fairing_height",
                "Family": "rocket_family",
            }
        )

        prep_summary = pd.DataFrame(
            {
                "metric": [
                    "total launches",
                    "missing launch timestamps",
                    "missing locations",
                    "location joins missing",
                    "unique raw locations",
                    "unique combined launch sites",
                    "mission aggregates available",
                    "config family available",
                ],
                "value": [
                    len(launch_df),
                    int(launch_df["launch_time_utc"].isna().sum()),
                    int(launch_df["Location"].isna().sum()),
                    int((~launch_df["location_joined"]).sum()),
                    int(launch_df["Location"].nunique()),
                    int(launch_df["Comb Launch Site"].nunique()),
                    int(launch_df["payload_count"].notna().sum()),
                    int(launch_df["rocket_family"].notna().sum()),
                ],
            }
        )

        prep_summary
        """
    ),
    md(
        """
        ## U.S. Scope for NOAA

        This notebook is intentionally limited to **U.S. launches** because the next step is to
        join launches to NOAA weather data using **launch date + ZIP code**. The raw launch data
        does not include ZIP codes, and that join strategy is only directly applicable to U.S.
        launch sites.
        """
    ),
    code(
        """
        us_launches = launch_df[launch_df["Country_Code"] == "US"].copy()

        us_summary = pd.DataFrame(
            {
                "metric": [
                    "U.S. launches",
                    "U.S. min launch date",
                    "U.S. max launch date",
                    "unique U.S. raw location strings",
                    "unique U.S. combined launch sites",
                ],
                "value": [
                    len(us_launches),
                    str(us_launches["launch_date"].min()),
                    str(us_launches["launch_date"].max()),
                    int(us_launches["Location"].nunique()),
                    int(us_launches["Comb Launch Site"].nunique()),
                ],
            }
        )

        us_summary
        """
    ),
    code(
        """
        us_status_counts = (
            us_launches["Launch Status"]
            .value_counts()
            .rename_axis("launch_status")
            .reset_index(name="launches")
        )

        us_status_counts
        """
    ),
    code(
        """
        us_by_decade = (
            us_launches.groupby("launch_decade")
            .size()
            .rename("launches")
            .reset_index()
            .sort_values("launch_decade")
        )

        us_by_decade
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        us_yearly_counts = us_launches.groupby("launch_year").size()
        us_yearly_counts.plot(ax=ax, linewidth=2, color="#d95f02")
        ax.set_title("U.S. Launches Per Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Launch count")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Launch-Site Consolidation

        Raw U.S. location strings contain many pad-level variants. For weather joins, the more
        useful unit is the facility group. The mapping below keeps Kennedy and Cape Canaveral
        separate because they need different practical ZIP proxies.
        """
    ),
    code(
        """
        def infer_us_facility_group(location: str) -> str:
            if "Kennedy Space Center" in location:
                return "Kennedy Space Center"
            if "Cape Canaveral SFS" in location:
                return "Cape Canaveral Space Force Station"
            if "Vandenberg SFB" in location:
                return "Vandenberg Space Force Base"
            if "Wallops Flight Facility" in location:
                return "Wallops Flight Facility"
            if "Pacific Spaceport Complex" in location:
                return "Pacific Spaceport Complex Alaska"
            if "Pacific Missile Range Facility" in location or "Kauai" in location:
                return "Pacific Missile Range Facility"
            if "Mojave Air and Space Port" in location:
                return "Mojave Air and Space Port"
            if "Edwards AFB" in location:
                return "Edwards Air Force Base"
            if "China Lake" in location:
                return "China Lake"
            return "Other U.S. site"


        us_launches["facility_group"] = us_launches["Location"].apply(infer_us_facility_group)

        us_site_summary = (
            us_launches.groupby("facility_group")
            .agg(
                launches=("Launch Id", "count"),
                raw_location_strings=("Location", "nunique"),
                first_launch=("launch_date", "min"),
                last_launch=("launch_date", "max"),
                success_rate=("Launch Status", lambda s: (s == "Success").mean()),
            )
            .sort_values("launches", ascending=False)
            .reset_index()
        )

        us_site_summary["success_rate"] = us_site_summary["success_rate"].map(lambda x: f"{x:.1%}")
        us_site_summary
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_df = (
            us_launches.groupby("facility_group")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="launches")
        )
        sns.barplot(data=plot_df, y="facility_group", x="launches", ax=ax, color="#1b9e77")
        ax.set_title("U.S. Launches by Facility Group")
        ax.set_xlabel("Launch count")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        monthly_counts = (
            us_launches.groupby("launch_month_name")
            .size()
            .reindex(month_order)
            .reset_index(name="launches")
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.barplot(data=monthly_counts, x="launch_month_name", y="launches", ax=ax, color="#7570b3")
        ax.set_title("U.S. Launches by Calendar Month")
        ax.set_xlabel("")
        ax.set_ylabel("Launch count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Candidate ZIP Codes for NOAA Pulls

        The raw data does not contain ZIP codes, so we derive them from latitude/longitude using
        `geopy` with Nominatim reverse geocoding. Where reverse geocoding does not return a
        postal code, the notebook uses your manual ZIP assignments so every U.S. facility has a
        usable NOAA lookup ZIP.
        """
    ),
    code(
        """
        facility_coords = (
            us_launches.groupby("facility_group")
            .agg(
                combined_site_lat=("Comb Launch Site Lat", "first"),
                combined_site_lon=("Comb Launch Site Lon", "first"),
                launches=("Launch Id", "count"),
            )
            .reset_index()
        )

        geolocator = Nominatim(user_agent="isye6740-rocket-eda")
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

        def reverse_lookup(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return {"geopy_postcode": None, "geopy_display_name": None}
            try:
                result = reverse((lat, lon), exactly_one=True, language="en")
                if result is None:
                    return {"geopy_postcode": None, "geopy_display_name": None}
                return {
                    "geopy_postcode": result.raw.get("address", {}).get("postcode"),
                    "geopy_display_name": result.address,
                }
            except Exception:
                return {"geopy_postcode": None, "geopy_display_name": None}


        geopy_results = facility_coords.apply(
            lambda row: pd.Series(reverse_lookup(row["combined_site_lat"], row["combined_site_lon"])),
            axis=1,
        )

        zip_reference = pd.concat([facility_coords, geopy_results], axis=1)
        zip_reference["geopy_found_zip"] = zip_reference["geopy_postcode"].notna()

        zip_reference[
            [
                "facility_group",
                "launches",
                "combined_site_lat",
                "combined_site_lon",
                "geopy_postcode",
                "geopy_found_zip",
            ]
        ].sort_values("launches", ascending=False)
        """
    ),
    code(
        """
        manual_zip_overrides = {
            "Cape Canaveral Space Force Station": "32925",
            "Kennedy Space Center": "32899",
            "Edwards Air Force Base": "93524",
            "Pacific Missile Range Facility": "96752",
            "Pacific Spaceport Complex Alaska": "99615",
        }

        manual_zip_override_df = pd.DataFrame(
            [
                {"facility_group": facility_group, "manual_zip_override": zip_code}
                for facility_group, zip_code in manual_zip_overrides.items()
            ]
        )

        zip_reference = zip_reference.merge(manual_zip_override_df, on="facility_group", how="left")
        zip_reference["candidate_zip"] = zip_reference["geopy_postcode"].fillna(zip_reference["manual_zip_override"])
        zip_reference["zip_source"] = zip_reference.apply(
            lambda row: (
                "geopy_reverse"
                if pd.notna(row["geopy_postcode"])
                else "manual_override"
                if pd.notna(row["manual_zip_override"])
                else "missing"
            ),
            axis=1,
        )
        zip_reference["zip_note"] = zip_reference.apply(
            lambda row: (
                f"Derived from geopy reverse lookup at ({row['combined_site_lat']:.6f}, {row['combined_site_lon']:.6f})."
                if row["zip_source"] == "geopy_reverse"
                else f"User-supplied ZIP override for facility_group={row['facility_group']}."
                if row["zip_source"] == "manual_override"
                else "ZIP missing."
            ),
            axis=1,
        )

        zip_reference[
            [
                "facility_group",
                "geopy_display_name",
                "geopy_postcode",
                "manual_zip_override",
                "candidate_zip",
                "zip_source",
                "zip_note",
            ]
        ].sort_values("facility_group")
        """
    ),
    code(
        """
        zip_reference_validation = zip_reference[
            [
                "facility_group",
                "geopy_display_name",
                "geopy_postcode",
                "manual_zip_override",
                "candidate_zip",
                "zip_source",
                "zip_note",
            ]
        ].sort_values("facility_group")

        zip_reference_validation
        """
    ),
    code(
        """
        zip_coverage_summary = pd.DataFrame(
            {
                "metric": [
                    "facility groups",
                    "facility groups with ZIP assigned",
                    "facility groups still missing ZIP",
                ],
                "value": [
                    int(zip_reference["facility_group"].nunique()),
                    int(zip_reference["candidate_zip"].notna().sum()),
                    int(zip_reference["candidate_zip"].isna().sum()),
                ],
            }
        )

        zip_coverage_summary
        """
    ),
    code(
        """

        us_zip_candidates = (
            us_launches[
                [
                    "Location",
                    "facility_group",
                    "Launch Site",
                    "Comb Launch Site",
                    "launch_date",
                    "Comb Launch Site Lat",
                    "Comb Launch Site Lon",
                ]
            ]
            .merge(
                zip_reference[
                    [
                        "facility_group",
                        "candidate_zip",
                        "zip_source",
                        "zip_note",
                    ]
                ],
                on="facility_group",
                how="left",
            )
            .rename(
                columns={
                    "Location": "raw_location",
                    "Launch Site": "launch_site",
                    "Comb Launch Site": "combined_launch_site",
                    "Comb Launch Site Lat": "combined_site_lat",
                    "Comb Launch Site Lon": "combined_site_lon",
                }
            )
        )

        zip_site_rollup = (
            us_zip_candidates.groupby(["facility_group", "candidate_zip", "zip_source", "zip_note"])
            .agg(
                launches=("raw_location", "count"),
                raw_location_strings=("raw_location", "nunique"),
                first_launch=("launch_date", "min"),
                last_launch=("launch_date", "max"),
            )
            .reset_index()
            .sort_values("launches", ascending=False)
        )

        zip_site_rollup
        """
    ),
    code(
        """
        raw_location_zip_map = (
            us_zip_candidates[
                [
                    "raw_location",
                    "facility_group",
                    "candidate_zip",
                    "zip_source",
                    "zip_note",
                    "launch_site",
                    "combined_launch_site",
                    "combined_site_lat",
                    "combined_site_lon",
                ]
            ]
            .drop_duplicates()
            .sort_values(["facility_group", "raw_location"])
            .reset_index(drop=True)
        )

        raw_location_zip_map
        """
    ),
    md(
        """
        ## Temporal Extraction Plan

        For NOAA pulls, the minimum workable extraction key is the launch date plus the resolved
        facility ZIP. This table gives the facility-level date windows that matter.
        """
    ),
    code(
        """
        noaa_pull_plan = (
            us_launches.groupby("facility_group")
            .agg(
                launches=("Launch Id", "count"),
                min_launch_date=("launch_date", "min"),
                max_launch_date=("launch_date", "max"),
                statuses=("Launch Status", lambda s: ", ".join(sorted(s.unique()))),
            )
            .reset_index()
            .merge(
                zip_reference[
                    [
                        "facility_group",
                        "candidate_zip",
                        "zip_source",
                        "zip_note",
                    ]
                ],
                on="facility_group",
                how="left",
            )
            .sort_values("launches", ascending=False)
        )

        noaa_pull_plan
        """
    ),
    md(
        """
        ## Data Quality Notes

        The launch-weather join is feasible, but a few issues should be handled deliberately:
        1. ZIP codes are not in the raw data and must be derived.
        2. Some ZIP codes come directly from `geopy` reverse geocoding, while others are supplied
           manually in `manual_zip_overrides` when geocoding does not return a postal code.
        3. `Locations.csv` appears to have at least one suspicious coordinate pair:
           `Pacific Spaceport Complex, Kodiak, Alaska` is associated with inland Alaska
           coordinates in the source file, so that site should be validated before any
           location-sensitive weather extraction.
        4. The dataset is entirely orbital launches, which simplifies the target population but
           means suborbital weather effects are out of scope here.
        """
    ),
    code(
        """
        suspicious_locations = locations[
            locations["Orig_Addr"].str.contains("Pacific Spaceport Complex", na=False)
        ][
            [
                "Orig_Addr",
                "Lat",
                "Lon",
                "Launch Site",
                "Comb Launch Site",
                "Comb Launch Site Lat",
                "Comb Launch Site Lon",
            ]
        ]

        suspicious_locations
        """
    ),
    md(
        """
        ## Final NOAA Pull Reference

        This is the main lookup table for weather extraction. Use one row per facility group to
        determine the ZIP code and inclusive date range to pull from NOAA.
        """
    ),
    code(
        """
        final_noaa_pull_reference = (
            noaa_pull_plan[
                [
                    "facility_group",
                    "candidate_zip",
                    "min_launch_date",
                    "max_launch_date",
                    "launches",
                    "zip_source",
                    "zip_note",
                ]
            ]
            .rename(
                columns={
                    "facility_group": "facility",
                    "candidate_zip": "zip_code",
                    "min_launch_date": "date_from",
                    "max_launch_date": "date_to",
                    "zip_source": "zip_assignment_method",
                    "zip_note": "zip_note",
                }
            )
            .sort_values(["facility"])
            .reset_index(drop=True)
        )

        final_noaa_pull_reference
        """
    ),
    md(
        """
        ## LCD Weather Merge

        The LCD weather files mix hourly, daily, and monthly rows. For the launch-weather merge,
        we keep only the strongest hourly observation at each timestamp, convert each launch time
        from UTC into the facility's **local standard time** (matching NOAA LCD conventions), and
        then match to the nearest hourly weather record within a 2-hour window.
        """
    ),
    code(
        """
        def clean_lcd_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(
                series.astype(str).str.extract(r"([-+]?[0-9]*\\.?[0-9]+)")[0],
                errors="coerce",
            )


        def load_best_hourly_weather(rel_path: str) -> pd.DataFrame:
            weather_raw = pd.read_csv(DATA_DIR / rel_path, low_memory=False)
            keep_cols = ["DATE", "REPORT_TYPE"] + [
                c
                for c in WEATHER_NUMERIC_COLUMNS + WEATHER_TEXT_COLUMNS + SHORT_DURATION_PRECIP_COLUMNS
                if c in weather_raw.columns
            ]
            weather = weather_raw[keep_cols].copy()

            for col in [c for c in WEATHER_NUMERIC_COLUMNS + SHORT_DURATION_PRECIP_COLUMNS if c in weather.columns]:
                weather[col] = clean_lcd_numeric(weather[col])

            weather["weather_obs_time_lstd"] = pd.to_datetime(weather["DATE"], errors="coerce")
            weather = weather.dropna(subset=["weather_obs_time_lstd"]).copy()

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

            present_weather = weather.get("HourlyPresentWeatherType", pd.Series("", index=weather.index)).fillna("").astype(str)
            sky_conditions = weather.get("HourlySkyConditions", pd.Series("", index=weather.index)).fillna("").astype(str)

            weather["present_weather_rain_flag"] = present_weather.str.contains(r"RA|DZ|SH", regex=True)
            weather["present_weather_fog_flag"] = present_weather.str.contains(r"FG|BR|HZ", regex=True)
            weather["present_weather_thunder_flag"] = present_weather.str.contains(r"TS", regex=True)
            weather["cloud_cover_broken_or_overcast_flag"] = sky_conditions.str.contains(r"BKN|OVC", regex=True)

            available_short_duration_cols = [c for c in SHORT_DURATION_PRECIP_COLUMNS if c in weather.columns]
            if available_short_duration_cols:
                weather["short_duration_precip_max"] = weather[available_short_duration_cols].max(axis=1, skipna=True)
            else:
                weather["short_duration_precip_max"] = float("nan")

            return weather


        weather_merges = []
        weather_coverage_rows = []

        for facility, (rel_path, utc_offset_hours) in WEATHER_FILE_MAP.items():
            facility_launches = us_launches.loc[us_launches["facility_group"] == facility].copy()
            facility_launches["launch_time_lstd"] = (
                facility_launches["launch_time_utc"] + pd.to_timedelta(utc_offset_hours, unit="h")
            ).dt.tz_localize(None)
            facility_launches = facility_launches.sort_values("launch_time_lstd")

            weather = load_best_hourly_weather(rel_path)

            merged = pd.merge_asof(
                facility_launches,
                weather,
                left_on="launch_time_lstd",
                right_on="weather_obs_time_lstd",
                direction="nearest",
                tolerance=pd.Timedelta("2h"),
            )

            merged["weather_matched"] = merged["weather_obs_time_lstd"].notna()
            merged["weather_time_diff_minutes"] = (
                (merged["launch_time_lstd"] - merged["weather_obs_time_lstd"]).dt.total_seconds().abs() / 60
            )

            weather_coverage_rows.append(
                {
                    "facility_group": facility,
                    "launches": len(facility_launches),
                    "matched_launches": int(merged["weather_matched"].sum()),
                    "match_rate": merged["weather_matched"].mean(),
                    "median_abs_diff_minutes": merged["weather_time_diff_minutes"].median(),
                    "weather_start_lstd": weather["weather_obs_time_lstd"].min(),
                    "weather_end_lstd": weather["weather_obs_time_lstd"].max(),
                }
            )

            weather_merges.append(merged)

        us_launch_weather = pd.concat(weather_merges, ignore_index=True)
        us_launch_weather["launch_outcome_group"] = us_launch_weather["Launch Status"].where(
            us_launch_weather["Launch Status"] == "Success",
            "Not Success",
        )
        us_launch_weather["launch_success_binary"] = (us_launch_weather["Launch Status"] == "Success").astype(int)
        us_launch_weather["launch_failure_binary"] = (us_launch_weather["Launch Status"] != "Success").astype(int)
        us_launch_weather["precip_positive_flag"] = us_launch_weather["HourlyPrecipitation"].fillna(0).gt(0)
        us_launch_weather["weather_type_reported_flag"] = (
            us_launch_weather["HourlyPresentWeatherType"].fillna("").astype(str).str.len().gt(0)
        )
        us_launch_weather["high_wind_flag"] = us_launch_weather["HourlyWindSpeed"].ge(15)
        us_launch_weather["low_visibility_flag"] = us_launch_weather["HourlyVisibility"].le(5)
        us_launch_weather["launch_year"] = us_launch_weather["launch_time_utc"].dt.year
        us_launch_weather["launch_decade"] = (us_launch_weather["launch_year"] // 10) * 10
        us_launch_weather["rocket_payload_leo"] = pd.to_numeric(
            us_launch_weather["Rocket Payload to LEO"], errors="coerce"
        )
        us_launch_weather["rocket_price_musd"] = pd.to_numeric(us_launch_weather["Rocket Price"], errors="coerce")
        us_launch_weather["rocket_price_adjusted_musd"] = pd.to_numeric(
            us_launch_weather["Rocket Price CPI Adjusted"], errors="coerce"
        )
        us_launch_weather["usd_per_kg_leo_adjusted"] = pd.to_numeric(
            us_launch_weather["USD/kg to LEO CPI Adjusted"], errors="coerce"
        )
        us_launch_weather["is_partial_failure"] = (us_launch_weather["Launch Status"] == "Partial Failure").astype(int)
        us_launch_weather["is_failure"] = (us_launch_weather["Launch Status"] == "Failure").astype(int)
        us_launch_weather["is_prelaunch_failure"] = (
            us_launch_weather["Launch Status"] == "Prelaunch Failure"
        ).astype(int)

        top_orgs = us_launch_weather["Rocket Organisation"].value_counts().head(8).index
        us_launch_weather["rocket_org_grouped"] = us_launch_weather["Rocket Organisation"].where(
            us_launch_weather["Rocket Organisation"].isin(top_orgs),
            "Other",
        )

        payload_bin_edges = [0, 500, 2000, 10000, 50000, 500000]
        us_launch_weather["payload_bin"] = pd.cut(
            us_launch_weather["rocket_payload_leo"],
            bins=payload_bin_edges,
            include_lowest=True,
        )

        weather_merge_coverage = (
            pd.DataFrame(weather_coverage_rows)
            .sort_values("matched_launches", ascending=False)
            .reset_index(drop=True)
        )

        weather_merge_coverage
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_df = weather_merge_coverage.sort_values("match_rate", ascending=False)
        sns.barplot(data=plot_df, x="match_rate", y="facility_group", ax=ax, color="#4c78a8")
        ax.set_title("Launch-Weather Merge Coverage by Facility")
        ax.set_xlabel("Share of launches matched to LCD weather within 2 hours")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Weather Feature Coverage

        Before comparing outcomes, it is important to see which LCD features are broadly available
        after the merge and which are sparse enough to be unreliable for first-pass modeling.
        """
    ),
    code(
        """
        matched_launch_weather = us_launch_weather.loc[us_launch_weather["weather_matched"]].copy()

        weather_feature_availability = pd.DataFrame(
            {
                "feature": WEATHER_NUMERIC_COLUMNS
                + WEATHER_TEXT_COLUMNS
                + [
                    "present_weather_rain_flag",
                    "present_weather_fog_flag",
                    "present_weather_thunder_flag",
                    "cloud_cover_broken_or_overcast_flag",
                    "short_duration_precip_max",
                ],
                "non_null_share": [
                    matched_launch_weather[col].notna().mean() if col in matched_launch_weather.columns else 0
                    for col in WEATHER_NUMERIC_COLUMNS
                    + WEATHER_TEXT_COLUMNS
                    + [
                        "present_weather_rain_flag",
                        "present_weather_fog_flag",
                        "present_weather_thunder_flag",
                        "cloud_cover_broken_or_overcast_flag",
                        "short_duration_precip_max",
                    ]
                ],
            }
        ).sort_values("non_null_share", ascending=False)

        weather_feature_availability
        """
    ),
    code(
        """
        facility_feature_availability = (
            matched_launch_weather.groupby("facility_group")[
                [c for c in WEATHER_NUMERIC_COLUMNS if c in matched_launch_weather.columns]
            ]
            .agg(lambda s: s.notna().mean())
            .T
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(facility_feature_availability, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Weather Feature Availability by Facility")
        plt.xlabel("Facility")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Launch Features in Scope

        Weather is only part of the story. The merged event table also retains launch-side
        variables such as operator, payload capacity, rocket price, and launch year so the EDA
        can distinguish weather patterns from baseline launch-vehicle differences.
        """
    ),
    code(
        """
        launch_feature_availability = pd.DataFrame(
            {
                "feature": [
                    "payload_count",
                    "mission_mass",
                    "mission_rows",
                    "rocket_payload_leo",
                    "rocket_price_musd",
                    "rocket_price_adjusted_musd",
                    "usd_per_kg_leo_adjusted",
                    "rocket_family",
                    "config_status",
                    "config_liftoff_thrust",
                    "config_payload_leo",
                    "config_payload_gto",
                    "config_stages",
                    "config_strap_ons",
                    "config_rocket_height",
                    "config_fairing_diameter",
                    "config_fairing_height",
                    "family_success_rate_pct",
                    "Rocket Organisation",
                    "Rocket Name",
                ],
                "non_null_share": [
                    us_launch_weather["payload_count"].notna().mean(),
                    us_launch_weather["mission_mass"].notna().mean(),
                    us_launch_weather["mission_rows"].notna().mean(),
                    us_launch_weather["rocket_payload_leo"].notna().mean(),
                    us_launch_weather["rocket_price_musd"].notna().mean(),
                    us_launch_weather["rocket_price_adjusted_musd"].notna().mean(),
                    us_launch_weather["usd_per_kg_leo_adjusted"].notna().mean(),
                    us_launch_weather["rocket_family"].notna().mean(),
                    us_launch_weather["config_status"].notna().mean(),
                    us_launch_weather["config_liftoff_thrust"].notna().mean(),
                    us_launch_weather["config_payload_leo"].notna().mean(),
                    us_launch_weather["config_payload_gto"].notna().mean(),
                    us_launch_weather["config_stages"].notna().mean(),
                    us_launch_weather["config_strap_ons"].notna().mean(),
                    us_launch_weather["config_rocket_height"].notna().mean(),
                    us_launch_weather["config_fairing_diameter"].notna().mean(),
                    us_launch_weather["config_fairing_height"].notna().mean(),
                    us_launch_weather["family_success_rate_pct"].notna().mean(),
                    us_launch_weather["Rocket Organisation"].notna().mean(),
                    us_launch_weather["Rocket Name"].notna().mean(),
                ],
            }
        )

        launch_feature_availability
        """
    ),
    code(
        """
        launch_feature_outcome_summary = []
        for feature in [
            "payload_count",
            "mission_mass",
            "rocket_payload_leo",
            "rocket_price_musd",
            "rocket_price_adjusted_musd",
            "usd_per_kg_leo_adjusted",
            "config_liftoff_thrust",
            "config_stages",
            "config_strap_ons",
            "family_success_rate_pct",
            "launch_year",
        ]:
            temp = (
                us_launch_weather.groupby("launch_outcome_group")[feature]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
            )
            temp["feature"] = feature
            launch_feature_outcome_summary.append(temp)

        launch_feature_outcome_summary = pd.concat(launch_feature_outcome_summary, ignore_index=True)
        launch_feature_outcome_summary
        """
    ),
    code(
        """
        rocket_org_outcomes = (
            us_launch_weather.groupby("rocket_org_grouped")
            .agg(
                launches=("Launch Id", "count"),
                success_rate=("launch_success_binary", "mean"),
            )
            .sort_values("launches", ascending=False)
            .reset_index()
        )

        rocket_org_outcomes
        """
    ),
    code(
        """
        rocket_family_outcomes = (
            us_launch_weather.groupby("rocket_family")
            .agg(
                launches=("Launch Id", "count"),
                success_rate=("launch_success_binary", "mean"),
            )
            .sort_values("launches", ascending=False)
            .head(15)
            .reset_index()
        )

        config_status_outcomes = (
            us_launch_weather.groupby("config_status")
            .agg(
                launches=("Launch Id", "count"),
                success_rate=("launch_success_binary", "mean"),
            )
            .sort_values("launches", ascending=False)
            .reset_index()
        )

        rocket_family_outcomes
        """
    ),
    code(
        """
        config_status_outcomes
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(
            data=us_launch_weather,
            x="launch_outcome_group",
            y="rocket_payload_leo",
            ax=axes[0],
            color="#f58518",
        )
        axes[0].set_title("Payload to LEO by Outcome")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Payload to LEO")
        axes[0].set_yscale("log")

        sns.boxplot(
            data=us_launch_weather,
            x="launch_outcome_group",
            y="rocket_price_adjusted_musd",
            ax=axes[1],
            color="#54a24b",
        )
        axes[1].set_title("CPI-Adjusted Rocket Price by Outcome")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Price (USD millions)")
        axes[1].set_yscale("log")

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        payload_success_profile = (
            us_launch_weather.dropna(subset=["payload_bin"])
            .groupby("payload_bin", observed=False)
            .agg(
                launches=("Launch Id", "count"),
                success_rate=("launch_success_binary", "mean"),
            )
            .reset_index()
        )

        fig, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=payload_success_profile, x="payload_bin", y="launches", ax=ax1, color="#eeca3b")
        ax1.set_ylabel("Launch count")
        ax1.set_xlabel("Payload to LEO bin")
        ax1.set_title("Launch Success Rate by Payload Bin")

        ax2 = ax1.twinx()
        sns.pointplot(data=payload_success_profile, x="payload_bin", y="success_rate", ax=ax2, color="#b279a2")
        ax2.set_ylabel("Success rate")
        ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.show()

        payload_success_profile
        """
    ),
    md(
        """
        ## Weather vs. Launch Outcome

        The dataset is highly imbalanced toward successful launches, so the comparisons below are
        descriptive EDA rather than causal conclusions. The goal is to surface patterns worth
        testing later in formal modeling.
        """
    ),
    code(
        """
        weather_by_outcome_summary = []
        for feature in [
            "HourlyAltimeterSetting",
            "HourlyDryBulbTemperature",
            "HourlyDewPointTemperature",
            "HourlyRelativeHumidity",
            "HourlyVisibility",
            "HourlySeaLevelPressure",
            "HourlyWetBulbTemperature",
            "HourlyWindSpeed",
            "short_duration_precip_max",
        ]:
            temp = (
                matched_launch_weather.groupby("launch_outcome_group")[feature]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
            )
            temp["feature"] = feature
            weather_by_outcome_summary.append(temp)

        weather_by_outcome_summary = pd.concat(weather_by_outcome_summary, ignore_index=True)
        weather_by_outcome_summary
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        feature_list = [
            "HourlyWindSpeed",
            "HourlyDryBulbTemperature",
            "HourlyRelativeHumidity",
            "HourlySeaLevelPressure",
        ]

        for ax, feature in zip(axes.flatten(), feature_list):
            sns.boxplot(
                data=matched_launch_weather,
                x="launch_outcome_group",
                y=feature,
                ax=ax,
                color="#72b7b2",
            )
            ax.set_title(feature)
            ax.set_xlabel("")
            ax.set_ylabel("")

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        outcome_indicator_summary = pd.DataFrame(
            {
                "metric": [
                    "precip_positive_share",
                    "weather_type_reported_share",
                    "high_wind_share",
                    "low_visibility_share",
                    "rain_code_share",
                    "fog_code_share",
                    "thunder_code_share",
                    "broken_or_overcast_share",
                ],
                "Not Success": [
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "precip_positive_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "weather_type_reported_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "high_wind_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "low_visibility_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "present_weather_rain_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "present_weather_fog_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "present_weather_thunder_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Not Success",
                        "cloud_cover_broken_or_overcast_flag",
                    ].mean(),
                ],
                "Success": [
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "precip_positive_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "weather_type_reported_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "high_wind_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "low_visibility_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "present_weather_rain_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "present_weather_fog_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "present_weather_thunder_flag",
                    ].mean(),
                    matched_launch_weather.loc[
                        matched_launch_weather["launch_outcome_group"] == "Success",
                        "cloud_cover_broken_or_overcast_flag",
                    ].mean(),
                ],
            }
        )

        outcome_indicator_summary
        """
    ),
    code(
        """
        wind_bin_edges = [0, 5, 10, 15, 20, 100]
        matched_launch_weather["wind_speed_bin"] = pd.cut(
            matched_launch_weather["HourlyWindSpeed"],
            bins=wind_bin_edges,
            include_lowest=True,
        )

        wind_success_profile = (
            matched_launch_weather.dropna(subset=["wind_speed_bin"])
            .groupby("wind_speed_bin", observed=False)
            .agg(
                launches=("Launch Id", "count"),
                success_rate=("launch_success_binary", "mean"),
            )
            .reset_index()
        )

        fig, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=wind_success_profile, x="wind_speed_bin", y="launches", ax=ax1, color="#a0cbe8")
        ax1.set_ylabel("Launch count")
        ax1.set_xlabel("Hourly wind speed bin")
        ax1.set_title("Launch Success Rate by Wind Speed Bin")

        ax2 = ax1.twinx()
        sns.pointplot(data=wind_success_profile, x="wind_speed_bin", y="success_rate", ax=ax2, color="#e45756")
        ax2.set_ylabel("Success rate")
        ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.show()

        wind_success_profile
        """
    ),
    code(
        """
        weather_corr_features = [
            "launch_success_binary",
            "launch_year",
            "rocket_payload_leo",
            "rocket_price_adjusted_musd",
            "HourlyDryBulbTemperature",
            "HourlyDewPointTemperature",
            "HourlyRelativeHumidity",
            "HourlyVisibility",
            "HourlySeaLevelPressure",
            "HourlyWindSpeed",
        ]
        weather_corr_matrix = matched_launch_weather[weather_corr_features].corr(numeric_only=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(weather_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Matrix for Launch Outcome, Launch Features, and Core Weather Features")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Write Derived Files

        These CSV outputs are ready to reuse in the next stage of the project.
        """
    ),
    code(
        """
        us_site_summary_export = (
            us_launches.groupby("facility_group")
            .agg(
                launches=("Launch Id", "count"),
                raw_location_strings=("Location", "nunique"),
                first_launch=("launch_date", "min"),
                last_launch=("launch_date", "max"),
                success_rate=("Launch Status", lambda s: (s == "Success").mean()),
            )
            .reset_index()
            .sort_values("launches", ascending=False)
        )

        us_site_summary_export.to_csv(OUTPUT_DIR / "us_launch_site_summary.csv", index=False)
        raw_location_zip_map.to_csv(OUTPUT_DIR / "us_launch_zip_candidates.csv", index=False)
        noaa_pull_plan.to_csv(OUTPUT_DIR / "noaa_pull_plan.csv", index=False)
        final_noaa_pull_reference.to_csv(OUTPUT_DIR / "final_noaa_pull_reference.csv", index=False)
        weather_merge_coverage.to_csv(OUTPUT_DIR / "weather_merge_coverage.csv", index=False)
        weather_feature_availability.to_csv(OUTPUT_DIR / "weather_feature_availability.csv", index=False)
        weather_by_outcome_summary.to_csv(OUTPUT_DIR / "weather_by_outcome_summary.csv", index=False)
        launch_feature_availability.to_csv(OUTPUT_DIR / "launch_feature_availability.csv", index=False)
        launch_feature_outcome_summary.to_csv(OUTPUT_DIR / "launch_feature_outcome_summary.csv", index=False)
        rocket_org_outcomes.to_csv(OUTPUT_DIR / "rocket_org_outcomes.csv", index=False)
        rocket_family_outcomes.to_csv(OUTPUT_DIR / "rocket_family_outcomes.csv", index=False)
        config_status_outcomes.to_csv(OUTPUT_DIR / "config_status_outcomes.csv", index=False)
        payload_success_profile.to_csv(OUTPUT_DIR / "payload_success_profile.csv", index=False)
        us_launch_weather.to_csv(OUTPUT_DIR / "us_launch_weather_merged.csv", index=False)

        pd.DataFrame(
            {
                "file": [
                    str(OUTPUT_DIR / "us_launch_site_summary.csv"),
                    str(OUTPUT_DIR / "us_launch_zip_candidates.csv"),
                    str(OUTPUT_DIR / "noaa_pull_plan.csv"),
                    str(OUTPUT_DIR / "final_noaa_pull_reference.csv"),
                    str(OUTPUT_DIR / "weather_merge_coverage.csv"),
                    str(OUTPUT_DIR / "weather_feature_availability.csv"),
                    str(OUTPUT_DIR / "weather_by_outcome_summary.csv"),
                    str(OUTPUT_DIR / "launch_feature_availability.csv"),
                    str(OUTPUT_DIR / "launch_feature_outcome_summary.csv"),
                    str(OUTPUT_DIR / "rocket_org_outcomes.csv"),
                    str(OUTPUT_DIR / "rocket_family_outcomes.csv"),
                    str(OUTPUT_DIR / "config_status_outcomes.csv"),
                    str(OUTPUT_DIR / "payload_success_profile.csv"),
                    str(OUTPUT_DIR / "us_launch_weather_merged.csv"),
                ]
            }
        )
        """
    ),
    md(
        """
        ## Bottom Line

        The U.S. portion of the launch dataset spans **1957-12-06 to 2021-12-21** and is
        dominated by two facilities:
        - Cape Canaveral / Kennedy
        - Vandenberg

        NOAA extraction can now use the final facility-level reference table written above instead
        of the raw pad-level location strings. With your manual ZIP entries added, every U.S.
        facility in scope now has a resolved ZIP/date pull specification. The notebook also
        produces an event-level launch-weather table that can be used directly for downstream
        modeling.
        """
    ),
]

nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.x"}

with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
