import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"c:\Users\djpsw\Georgia Tech\ISYE6740\final_project")
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "derived"
OUTPUT_DIR.mkdir(exist_ok=True)


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [(line + "\n") for line in text.strip().splitlines()],
    }


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [(line + "\n") for line in text.strip().splitlines()],
    }


def parse_numeric_text(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0],
        errors="coerce",
    )


def infer_us_facility_group(location: str) -> str:
    location = "" if pd.isna(location) else str(location)
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


def clean_lcd_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0],
        errors="coerce",
    )


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
    weather["short_duration_precip_max"] = (
        weather[available_short_duration_cols].max(axis=1, skipna=True)
        if available_short_duration_cols
        else float("nan")
    )
    return weather


def add_prelaunch_history(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    valid = df[group_col].notna()
    grp = df.loc[valid].groupby(group_col, sort=False)
    df.loc[valid, f"{prefix}_prior_launches"] = grp.cumcount()
    df.loc[valid, f"{prefix}_prior_successes"] = grp["launch_success_binary"].cumsum() - df.loc[valid, "launch_success_binary"]
    df[f"{prefix}_success_rate_prelaunch"] = df[f"{prefix}_prior_successes"] / df[f"{prefix}_prior_launches"].replace(0, np.nan)
    first_time = grp["launch_time_utc"].transform("min")
    prev_time = grp["launch_time_utc"].shift(1)
    df.loc[valid, f"{prefix}_years_since_first_launch"] = (
        (df.loc[valid, "launch_time_utc"] - first_time).dt.total_seconds() / (365.25 * 24 * 3600)
    )
    df.loc[valid, f"days_since_previous_launch_{prefix}"] = (
        (df.loc[valid, "launch_time_utc"] - prev_time).dt.total_seconds() / (24 * 3600)
    )
    return df


def engineer_merged_features(us_launch_weather: pd.DataFrame) -> pd.DataFrame:
    us_launch_weather = us_launch_weather.sort_values("launch_time_utc").reset_index(drop=True).copy()
    us_launch_weather["launch_outcome_group"] = us_launch_weather["Launch Status"].where(
        us_launch_weather["Launch Status"] == "Success", "Not Success"
    )
    us_launch_weather["launch_success_binary"] = (us_launch_weather["Launch Status"] == "Success").astype(int)
    us_launch_weather["launch_failure_binary"] = (us_launch_weather["Launch Status"] != "Success").astype(int)
    us_launch_weather["is_partial_failure"] = (us_launch_weather["Launch Status"] == "Partial Failure").astype(int)
    us_launch_weather["is_failure"] = (us_launch_weather["Launch Status"] == "Failure").astype(int)
    us_launch_weather["is_prelaunch_failure"] = (us_launch_weather["Launch Status"] == "Prelaunch Failure").astype(int)

    us_launch_weather["rocket_payload_leo"] = pd.to_numeric(us_launch_weather["Rocket Payload to LEO"], errors="coerce")
    us_launch_weather["rocket_price_musd"] = pd.to_numeric(us_launch_weather["Rocket Price"], errors="coerce")
    us_launch_weather["rocket_price_adjusted_musd"] = pd.to_numeric(
        us_launch_weather["Rocket Price CPI Adjusted"], errors="coerce"
    )
    us_launch_weather["usd_per_kg_leo_adjusted"] = pd.to_numeric(
        us_launch_weather["USD/kg to LEO CPI Adjusted"], errors="coerce"
    )

    top_orgs = us_launch_weather["Rocket Organisation"].value_counts().head(8).index
    us_launch_weather["rocket_org_grouped"] = us_launch_weather["Rocket Organisation"].where(
        us_launch_weather["Rocket Organisation"].isin(top_orgs), "Other"
    )
    payload_bin_edges = [0, 500, 2000, 10000, 50000, 500000]
    us_launch_weather["payload_bin"] = pd.cut(
        us_launch_weather["rocket_payload_leo"], bins=payload_bin_edges, include_lowest=True
    )

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    us_launch_weather["launch_season"] = us_launch_weather["launch_month"].map(season_map)
    us_launch_weather["launch_hour_local"] = us_launch_weather["launch_time_lstd"].dt.hour
    us_launch_weather["launch_hour_bin"] = pd.cut(
        us_launch_weather["launch_hour_local"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Overnight", "Morning", "Afternoon", "Evening"],
    )
    us_launch_weather["era_group"] = pd.cut(
        us_launch_weather["launch_year"],
        bins=[1950, 1979, 1999, 2025],
        labels=["Early space age (1951-1979)", "Transition era (1980-1999)", "Modern era (2000-2024)"],
        include_lowest=True,
    )

    us_launch_weather["precip_positive_flag"] = us_launch_weather["HourlyPrecipitation"].fillna(0).gt(0)
    us_launch_weather["weather_type_reported_flag"] = (
        us_launch_weather["HourlyPresentWeatherType"].fillna("").astype(str).str.len().gt(0)
    )
    us_launch_weather["high_wind_flag"] = us_launch_weather["HourlyWindSpeed"].ge(15)
    us_launch_weather["low_visibility_flag"] = us_launch_weather["HourlyVisibility"].le(5)
    us_launch_weather["high_humidity_flag"] = us_launch_weather["HourlyRelativeHumidity"].ge(80)
    us_launch_weather["dewpoint_depression"] = (
        us_launch_weather["HourlyDryBulbTemperature"] - us_launch_weather["HourlyDewPointTemperature"]
    )
    us_launch_weather["has_any_precip_signal"] = (
        us_launch_weather[["precip_positive_flag", "present_weather_rain_flag"]]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    )
    us_launch_weather["high_wind_and_low_visibility_flag"] = (
        us_launch_weather["high_wind_flag"].fillna(False).astype(bool)
        & us_launch_weather["low_visibility_flag"].fillna(False).astype(bool)
    )
    us_launch_weather["high_wind_and_high_humidity_flag"] = (
        us_launch_weather["high_wind_flag"].fillna(False).astype(bool)
        & us_launch_weather["high_humidity_flag"].fillna(False).astype(bool)
    )
    us_launch_weather["rain_and_low_visibility_flag"] = (
        us_launch_weather["has_any_precip_signal"].fillna(False).astype(bool)
        & us_launch_weather["low_visibility_flag"].fillna(False).astype(bool)
    )
    us_launch_weather["wind_x_visibility"] = us_launch_weather["HourlyWindSpeed"] * us_launch_weather["HourlyVisibility"]
    us_launch_weather["wind_x_humidity"] = us_launch_weather["HourlyWindSpeed"] * us_launch_weather["HourlyRelativeHumidity"]

    us_launch_weather["weather_match_quality_bin"] = pd.cut(
        us_launch_weather["weather_time_diff_minutes"],
        bins=[0, 15, 30, 60, 120],
        labels=["0-15 min", "15-30 min", "30-60 min", "60-120 min"],
        include_lowest=True,
    )
    us_launch_weather["multi_payload_flag"] = us_launch_weather["payload_count"].fillna(0).gt(1)
    us_launch_weather["rocket_price_adjusted_missing_flag"] = us_launch_weather["rocket_price_adjusted_musd"].isna()
    us_launch_weather["usd_per_kg_leo_adjusted_missing_flag"] = us_launch_weather["usd_per_kg_leo_adjusted"].isna()
    us_launch_weather["config_fairing_diameter_missing_flag"] = us_launch_weather["config_fairing_diameter"].isna()
    us_launch_weather["config_fairing_height_missing_flag"] = us_launch_weather["config_fairing_height"].isna()
    us_launch_weather["HourlyVisibility_missing_flag"] = us_launch_weather["HourlyVisibility"].isna()
    us_launch_weather["HourlyAltimeterSetting_missing_flag"] = us_launch_weather["HourlyAltimeterSetting"].isna()
    us_launch_weather["HourlyWetBulbTemperature_missing_flag"] = us_launch_weather["HourlyWetBulbTemperature"].isna()

    for group_col, prefix in [
        ("rocket_family", "family"),
        ("Rocket Organisation", "org"),
        ("Rocket Name", "config"),
        ("facility_group", "site"),
    ]:
        us_launch_weather = add_prelaunch_history(us_launch_weather, group_col, prefix)

    us_launch_weather["site_launches_so_far"] = us_launch_weather["site_prior_launches"]
    us_launch_weather["family_launches_so_far"] = us_launch_weather["family_prior_launches"]
    us_launch_weather["config_launches_so_far"] = us_launch_weather["config_prior_launches"]
    us_launch_weather["org_launches_so_far"] = us_launch_weather["org_prior_launches"]

    for raw_col, new_col in [
        ("HourlyWindSpeed", "site_wind_speed_z"),
        ("HourlyVisibility", "site_visibility_z"),
        ("HourlyRelativeHumidity", "site_humidity_z"),
        ("dewpoint_depression", "site_dewpoint_depression_z"),
        ("HourlyAltimeterSetting", "site_altimeter_setting_z"),
    ]:
        site_mean = us_launch_weather.groupby("facility_group")[raw_col].transform("mean")
        site_std = us_launch_weather.groupby("facility_group")[raw_col].transform("std")
        us_launch_weather[new_col] = (us_launch_weather[raw_col] - site_mean) / site_std.replace(0, np.nan)

    return us_launch_weather


# Build updated derived merged file and supporting prep summaries.
companies = pd.read_csv(DATA_DIR / "Companies.csv")
configs = pd.read_csv(DATA_DIR / "Configs.csv")
families = pd.read_csv(DATA_DIR / "Families.csv")
launches = pd.read_csv(DATA_DIR / "Launches.csv")
locations = pd.read_csv(DATA_DIR / "Locations.csv")
missions = pd.read_csv(DATA_DIR / "Missions.csv")

launch_df = launches.copy()
launch_df["launch_time_utc"] = pd.to_datetime(launch_df["Launch Time"], utc=True, errors="coerce")
launch_df["launch_date"] = launch_df["launch_time_utc"].dt.date
launch_df["launch_year"] = launch_df["launch_time_utc"].dt.year
launch_df["launch_month"] = launch_df["launch_time_utc"].dt.month
launch_df["launch_month_name"] = launch_df["launch_time_utc"].dt.month_name()
launch_df["launch_decade"] = (launch_df["launch_year"] // 10) * 10
mission_agg = (
    missions.groupby("Launch Id")
    .agg(payload_count=("Payloads", "sum"), mission_mass=("Mass", "sum"), mission_rows=("No", "count"))
    .reset_index()
)
config_features = configs.merge(
    families[["Family Id", "Family", "Success Rate"]], on="Family Id", how="left"
).copy()
for col in [
    "Liftoff Thrust", "Payload to LEO", "Payload to GTO", "Stages", "Strap-ons",
    "Rocket Height", "Fairing Diameter", "Fairing Height",
]:
    if col in config_features.columns:
        config_features[col] = parse_numeric_text(config_features[col])
config_features["family_success_rate_pct"] = pd.to_numeric(
    config_features["Success Rate"].astype(str).str.rstrip("%"), errors="coerce"
)
launch_df = launch_df.merge(
    locations[
        [
            "Orig_Addr", "Country", "Country_Code", "Operator", "Launch Site", "Comb Launch Site",
            "Lat", "Lon", "Comb Launch Site Lat", "Comb Launch Site Lon",
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
            "Config", "Status", "Liftoff Thrust", "Payload to LEO", "Payload to GTO",
            "Stages", "Strap-ons", "Rocket Height", "Fairing Diameter", "Fairing Height",
            "Family", "family_success_rate_pct",
        ]
    ],
    left_on="Rocket Name",
    right_on="Config",
    how="left",
)
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
us_launches = launch_df.loc[launch_df["Country_Code"] == "US"].copy()
us_launches["facility_group"] = us_launches["Location"].apply(infer_us_facility_group)

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

us_launch_weather = engineer_merged_features(pd.concat(weather_merges, ignore_index=True))
weather_merge_coverage = pd.DataFrame(weather_coverage_rows).sort_values("matched_launches", ascending=False).reset_index(drop=True)
matched_launch_weather = us_launch_weather.loc[us_launch_weather["weather_matched"]].copy()
weather_feature_availability = pd.DataFrame(
    {
        "feature": WEATHER_NUMERIC_COLUMNS + WEATHER_TEXT_COLUMNS + [
            "present_weather_rain_flag", "present_weather_fog_flag",
            "present_weather_thunder_flag", "cloud_cover_broken_or_overcast_flag",
            "short_duration_precip_max",
        ],
        "non_null_share": [
            matched_launch_weather[col].notna().mean() if col in matched_launch_weather.columns else 0
            for col in WEATHER_NUMERIC_COLUMNS + WEATHER_TEXT_COLUMNS + [
                "present_weather_rain_flag", "present_weather_fog_flag",
                "present_weather_thunder_flag", "cloud_cover_broken_or_overcast_flag",
                "short_duration_precip_max",
            ]
        ],
    }
).sort_values("non_null_share", ascending=False)
facility_feature_availability = (
    matched_launch_weather.groupby("facility_group")[
        [c for c in WEATHER_NUMERIC_COLUMNS if c in matched_launch_weather.columns]
    ]
    .agg(lambda s: s.notna().mean())
    .T
    .reset_index()
    .rename(columns={"index": "feature"})
)
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

us_launch_weather.to_csv(OUTPUT_DIR / "us_launch_weather_merged.csv", index=False)
weather_merge_coverage.to_csv(OUTPUT_DIR / "weather_merge_coverage.csv", index=False)
weather_feature_availability.to_csv(OUTPUT_DIR / "weather_feature_availability.csv", index=False)
facility_feature_availability.to_csv(OUTPUT_DIR / "facility_weather_feature_availability.csv", index=False)
us_site_summary_export.to_csv(OUTPUT_DIR / "us_launch_site_summary.csv", index=False)


def build_candidate_markdown(model_feature_tiers: pd.DataFrame, missingness_signal_summary: pd.DataFrame) -> str:
    defs = {
        "family_success_rate_prelaunch": (
            "Pre-launch historical success rate of the rocket family.",
            "Time-safe replacement for family history; should preserve strong maturity signal without leakage.",
            "Still depends on enough prior family launches to be stable.",
        ),
        "org_success_rate_prelaunch": (
            "Pre-launch historical success rate of the rocket organisation.",
            "Captures operator/program maturity beyond a single family.",
            "May partly proxy for era and site mix.",
        ),
        "config_success_rate_prelaunch": (
            "Pre-launch historical success rate of the exact rocket configuration.",
            "Potentially strong if exact-configuration history matters for reliability.",
            "Can be sparse for rare or new configurations.",
        ),
        "HourlyWindSpeed": (
            "Matched hourly wind speed near launch time.",
            "Best weather-side variable so far in pooled and adjusted EDA.",
            "Can still reflect site climatology.",
        ),
        "site_wind_speed_z": (
            "Wind speed standardized relative to the typical level at that site.",
            "More comparable across facilities than raw wind speed; directly aligned with site-conditional weather risk.",
            "Depends on stable site-level weather baselines.",
        ),
        "config_rocket_height": (
            "Overall rocket height for the matched configuration.",
            "Strong proxy for vehicle class and architecture.",
            "Likely correlated with other size/capability variables.",
        ),
        "config_liftoff_thrust": (
            "Rocket liftoff thrust for the matched configuration.",
            "Captures vehicle scale and mission envelope.",
            "Can overlap with payload and family variables.",
        ),
        "rocket_payload_leo": (
            "Payload-to-LEO value reported in the launch table.",
            "Useful capability proxy with persistent adjusted separation.",
            "Only moderate coverage.",
        ),
        "mission_mass": (
            "Total aggregated mission mass for the launch.",
            "Adds mission-scale context beyond nominal rocket capability.",
            "Depends on mission-table completeness.",
        ),
        "config_strap_ons": (
            "Number of strap-on boosters in the configuration.",
            "High coverage and useful architecture/complexity proxy.",
            "May mostly proxy for family identity.",
        ),
        "config_stages": (
            "Number of stages in the configuration.",
            "Simple high-coverage architecture variable.",
            "May mainly encode family/config type.",
        ),
        "site_visibility_z": (
            "Visibility standardized relative to the typical level at that site.",
            "More interpretable across facilities than raw visibility and tied to site-conditional weather risk.",
            "Coverage is still uneven by site.",
        ),
        "HourlyVisibility": (
            "Matched hourly visibility near launch time.",
            "Operationally intuitive secondary weather feature.",
            "Coverage varies sharply by facility.",
        ),
        "high_wind_and_low_visibility_flag": (
            "Flag for launches with both high wind and low visibility.",
            "Useful interaction candidate because combined bad conditions may matter more than each alone.",
            "Interaction cells can be sparse.",
        ),
        "family_launches_so_far": (
            "Number of prior launches observed for the rocket family.",
            "Direct maturity/count feature that may generalize better than raw success rate alone.",
            "Closely related to historical family effects.",
        ),
        "days_since_previous_launch_site": (
            "Days since the previous launch at the same site.",
            "Potential cadence/operations feature for launch-site readiness and tempo.",
            "Interpretation may vary across programs and eras.",
        ),
        "days_since_previous_launch_family": (
            "Days since the previous launch for the same rocket family.",
            "Potential cadence feature for family-level operational tempo.",
            "Can be unstable for infrequent families.",
        ),
    }
    candidate_df = model_feature_tiers.loc[
        model_feature_tiers["modeling_tier"].isin(["core modeling candidate", "secondary candidate"])
    ].copy()
    candidate_df = candidate_df.merge(
        missingness_signal_summary[["feature", "missing_rate_gap_not_success_minus_success"]],
        on="feature",
        how="left",
    )
    headers = ["feature", "modeling_tier", "meaning", "why_candidate", "eda_evidence", "caution"]
    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in candidate_df.sort_values(["modeling_tier", "abs_within_site_era_z_gap"], ascending=[True, False]).iterrows():
        meaning, why_candidate, caution = defs.get(row["feature"], ("Definition not yet added.", "EDA-backed rationale not yet written.", ""))
        eda = (
            f"cov={row['non_null_share']:.2f}; pooled={row['abs_pooled_corr_with_success']:.3f}; "
            f"adjusted={row['abs_within_site_era_z_gap']:.3f}"
        )
        if pd.notna(row["missing_rate_gap_not_success_minus_success"]):
            eda += f"; miss_gap={row['missing_rate_gap_not_success_minus_success']:.3f}"
        vals = [row["feature"], row["modeling_tier"], meaning, why_candidate, eda, caution]
        vals = [str(v).replace("|", "\\|") for v in vals]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


analysis_header = """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid", palette="deep")

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "derived"
us_launch_weather = pd.read_csv(
    OUTPUT_DIR / "us_launch_weather_merged.csv",
    parse_dates=["launch_time_utc", "launch_time_lstd", "weather_obs_time_lstd"],
)
matched_launch_weather = us_launch_weather.loc[us_launch_weather["weather_matched"]].copy()
weather_merge_coverage = pd.read_csv(OUTPUT_DIR / "weather_merge_coverage.csv")
weather_feature_availability = pd.read_csv(OUTPUT_DIR / "weather_feature_availability.csv")
facility_weather_feature_availability = pd.read_csv(OUTPUT_DIR / "facility_weather_feature_availability.csv")


def grouped_success_profile(df: pd.DataFrame, feature: str, min_launches: int = 1) -> pd.DataFrame:
    profile = (
        df.groupby(feature, dropna=False, observed=False)
        .agg(launches=("Launch Id", "count"), success_rate=("launch_success_binary", "mean"))
        .reset_index()
    )
    return profile.loc[profile["launches"] >= min_launches].copy()


def binned_success_profile(
    df: pd.DataFrame,
    feature: str,
    *,
    quantiles: int | None = None,
    bin_edges: list[float] | None = None,
    labels: list[str] | None = None,
    min_launches: int = 10,
) -> pd.DataFrame:
    temp = df[[feature, "Launch Id", "launch_success_binary"]].dropna().copy()
    if quantiles is not None:
        temp["feature_bin"] = pd.qcut(temp[feature], q=quantiles, duplicates="drop")
    elif bin_edges is not None:
        temp["feature_bin"] = pd.cut(temp[feature], bins=bin_edges, labels=labels, include_lowest=True)
    else:
        raise ValueError("Specify either quantiles or bin_edges.")
    profile = (
        temp.groupby("feature_bin", observed=False)
        .agg(launches=("Launch Id", "count"), success_rate=("launch_success_binary", "mean"))
        .reset_index()
    )
    profile["feature_bin"] = profile["feature_bin"].astype(str)
    return profile.loc[profile["launches"] >= min_launches].copy()
"""

feature_screen_code = """
def grouped_success_profile(df: pd.DataFrame, feature: str, min_launches: int = 1) -> pd.DataFrame:
    profile = (
        df.groupby(feature, dropna=False, observed=False)
        .agg(launches=("Launch Id", "count"), success_rate=("launch_success_binary", "mean"))
        .reset_index()
    )
    return profile.loc[profile["launches"] >= min_launches].copy()


def binned_success_profile(
    df: pd.DataFrame,
    feature: str,
    *,
    quantiles: int | None = None,
    bin_edges: list[float] | None = None,
    labels: list[str] | None = None,
    min_launches: int = 10,
) -> pd.DataFrame:
    temp = df[[feature, "Launch Id", "launch_success_binary"]].dropna().copy()
    if quantiles is not None:
        temp["feature_bin"] = pd.qcut(temp[feature], q=quantiles, duplicates="drop")
    elif bin_edges is not None:
        temp["feature_bin"] = pd.cut(temp[feature], bins=bin_edges, labels=labels, include_lowest=True)
    else:
        raise ValueError("Specify either quantiles or bin_edges.")
    profile = (
        temp.groupby("feature_bin", observed=False)
        .agg(launches=("Launch Id", "count"), success_rate=("launch_success_binary", "mean"))
        .reset_index()
    )
    profile["feature_bin"] = profile["feature_bin"].astype(str)
    return profile.loc[profile["launches"] >= min_launches].copy()


feature_screen_candidates = [
    ("family_success_rate_prelaunch", "launch"),
    ("org_success_rate_prelaunch", "launch"),
    ("config_success_rate_prelaunch", "launch"),
    ("family_launches_so_far", "launch"),
    ("site_launches_so_far", "launch"),
    ("days_since_previous_launch_site", "launch"),
    ("days_since_previous_launch_family", "launch"),
    ("rocket_payload_leo", "launch"),
    ("mission_mass", "launch"),
    ("config_liftoff_thrust", "launch"),
    ("config_rocket_height", "launch"),
    ("config_stages", "launch"),
    ("config_strap_ons", "launch"),
    ("config_payload_leo", "launch"),
    ("HourlyWindSpeed", "weather"),
    ("site_wind_speed_z", "weather"),
    ("HourlyVisibility", "weather"),
    ("site_visibility_z", "weather"),
    ("HourlyAltimeterSetting", "weather"),
    ("high_wind_and_low_visibility_flag", "weather"),
    ("high_wind_and_high_humidity_flag", "weather"),
    ("rain_and_low_visibility_flag", "weather"),
]

feature_screen_rows = []
for feature, feature_group in feature_screen_candidates:
    base_df = matched_launch_weather if feature_group == "weather" else us_launch_weather
    temp = base_df[["facility_group", "era_group", "launch_success_binary", feature]].dropna().copy()
    if len(temp) <= 5:
        continue
    pooled_corr = temp[["launch_success_binary", feature]].corr(numeric_only=True).iloc[0, 1]
    temp["site_era_mean"] = temp.groupby(["facility_group", "era_group"])[feature].transform("mean")
    temp["site_era_std"] = temp.groupby(["facility_group", "era_group"])[feature].transform("std")
    temp = temp.loc[temp["site_era_std"].fillna(0) > 0].copy()
    if len(temp) > 5:
        temp["site_era_z"] = (temp[feature] - temp["site_era_mean"]) / temp["site_era_std"]
        success_mean = temp.loc[temp["launch_success_binary"] == 1, "site_era_z"].mean()
        failure_mean = temp.loc[temp["launch_success_binary"] == 0, "site_era_z"].mean()
        adjusted_gap = success_mean - failure_mean
    else:
        adjusted_gap = pd.NA
    feature_screen_rows.append(
        {
            "feature": feature,
            "feature_group": feature_group,
            "non_null_share": base_df[feature].notna().mean(),
            "sample_size_used": base_df[feature].notna().sum(),
            "pooled_corr_with_success": pooled_corr,
            "abs_pooled_corr_with_success": abs(pooled_corr),
            "within_site_era_z_gap_success_minus_failure": adjusted_gap,
            "abs_within_site_era_z_gap": abs(adjusted_gap) if pd.notna(adjusted_gap) else pd.NA,
        }
    )

feature_screening_summary = pd.DataFrame(feature_screen_rows).sort_values(
    ["abs_within_site_era_z_gap", "non_null_share"], ascending=[False, False]
)
"""

missingness_code = """
missingness_candidates = [
    "rocket_price_adjusted_missing_flag",
    "usd_per_kg_leo_adjusted_missing_flag",
    "config_fairing_diameter_missing_flag",
    "config_fairing_height_missing_flag",
    "HourlyVisibility_missing_flag",
    "HourlyAltimeterSetting_missing_flag",
    "HourlyWetBulbTemperature_missing_flag",
]

missingness_rows = []
for feature in missingness_candidates:
    base_df = matched_launch_weather if feature.startswith("Hourly") else us_launch_weather
    indicator = base_df[feature].fillna(False).astype(bool)
    missingness_rows.append(
        {
            "feature": feature,
            "feature_group": "weather" if feature.startswith("Hourly") else "launch",
            "missing_share": indicator.mean(),
            "success_missing_rate": indicator.loc[base_df["launch_outcome_group"] == "Success"].mean(),
            "not_success_missing_rate": indicator.loc[base_df["launch_outcome_group"] == "Not Success"].mean(),
            "missing_rate_gap_not_success_minus_success": (
                indicator.loc[base_df["launch_outcome_group"] == "Not Success"].mean()
                - indicator.loc[base_df["launch_outcome_group"] == "Success"].mean()
            ),
        }
    )

missingness_signal_summary = pd.DataFrame(missingness_rows).sort_values(
    "missing_rate_gap_not_success_minus_success",
    key=lambda s: s.abs(),
    ascending=False,
)
"""

model_tiers_code = """
feature_tier_rows = []
for _, row in feature_screening_summary.iterrows():
    coverage = row["non_null_share"]
    adjusted_gap = row["abs_within_site_era_z_gap"] if pd.notna(row["abs_within_site_era_z_gap"]) else 0
    pooled_gap = row["abs_pooled_corr_with_success"] if pd.notna(row["abs_pooled_corr_with_success"]) else 0
    if coverage >= 0.65 and adjusted_gap >= 0.12:
        tier = "core modeling candidate"
    elif coverage >= 0.40 and max(adjusted_gap, pooled_gap) >= 0.10:
        tier = "secondary candidate"
    else:
        tier = "exploratory / sparse candidate"
    feature_tier_rows.append(
        {
            "feature": row["feature"],
            "feature_group": row["feature_group"],
            "non_null_share": coverage,
            "abs_pooled_corr_with_success": pooled_gap,
            "abs_within_site_era_z_gap": adjusted_gap,
            "modeling_tier": tier,
        }
    )

model_feature_tiers = pd.DataFrame(feature_tier_rows).sort_values(
    ["modeling_tier", "abs_within_site_era_z_gap", "non_null_share"],
    ascending=[True, False, False],
)
"""


# Build analysis summaries for markdown table and exports.
exec_env = {
    "pd": pd,
    "np": np,
    "us_launch_weather": us_launch_weather.copy(),
    "matched_launch_weather": matched_launch_weather.copy(),
}
exec(feature_screen_code, exec_env)
exec(missingness_code, exec_env)
exec(model_tiers_code, exec_env)
feature_screening_summary = exec_env["feature_screening_summary"]
missingness_signal_summary = exec_env["missingness_signal_summary"]
model_feature_tiers = exec_env["model_feature_tiers"]
candidate_table_md = build_candidate_markdown(model_feature_tiers, missingness_signal_summary)

feature_screening_summary.to_csv(OUTPUT_DIR / "feature_screening_summary.csv", index=False)
missingness_signal_summary.to_csv(OUTPUT_DIR / "missingness_signal_summary.csv", index=False)
model_feature_tiers.to_csv(OUTPUT_DIR / "model_feature_tiers.csv", index=False)


data_prep_cells = [
    md(
        """
# Rocket Launch Data Preparation

This notebook handles raw-data preparation and feature engineering for the project. It loads the launch, mission, family, configuration, location, and weather files; creates the modeling features used in downstream analysis; and writes the prepared event table to `data/derived/`.
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

WEATHER_NUMERIC_COLUMNS = %s
WEATHER_TEXT_COLUMNS = %s
SHORT_DURATION_PRECIP_COLUMNS = %s
WEATHER_FILE_MAP = %s
"""
        % (repr(WEATHER_NUMERIC_COLUMNS), repr(WEATHER_TEXT_COLUMNS), repr(SHORT_DURATION_PRECIP_COLUMNS), repr(WEATHER_FILE_MAP))
    ),
    md("## 1. Load Source Tables"),
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
        "dataset": ["Companies", "Configs", "Families", "Launches", "Locations", "Missions"],
        "rows": [len(companies), len(configs), len(families), len(launches), len(locations), len(missions)],
        "columns": [companies.shape[1], configs.shape[1], families.shape[1], launches.shape[1], locations.shape[1], missions.shape[1]],
    }
)
dataset_shapes
"""
    ),
    md("## 2. Launch-Level Preparation"),
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
    .agg(payload_count=("Payloads", "sum"), mission_mass=("Mass", "sum"), mission_rows=("No", "count"))
    .reset_index()
)

config_features = configs.merge(
    families[["Family Id", "Family", "Success Rate"]],
    on="Family Id",
    how="left",
).copy()

for col in [
    "Liftoff Thrust", "Payload to LEO", "Payload to GTO", "Stages",
    "Strap-ons", "Rocket Height", "Fairing Diameter", "Fairing Height",
]:
    if col in config_features.columns:
        config_features[col] = parse_numeric_text(config_features[col])

config_features["family_success_rate_pct"] = pd.to_numeric(
    config_features["Success Rate"].astype(str).str.rstrip("%"),
    errors="coerce",
)

launch_df = launch_df.merge(
    locations[
        [
            "Orig_Addr", "Country", "Country_Code", "Operator", "Launch Site",
            "Comb Launch Site", "Lat", "Lon", "Comb Launch Site Lat", "Comb Launch Site Lon",
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
            "Config", "Status", "Liftoff Thrust", "Payload to LEO", "Payload to GTO",
            "Stages", "Strap-ons", "Rocket Height", "Fairing Diameter", "Fairing Height",
            "Family", "family_success_rate_pct",
        ]
    ],
    left_on="Rocket Name",
    right_on="Config",
    how="left",
)
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
            "total launches", "missing launch timestamps", "missing locations",
            "unique raw locations", "unique combined launch sites", "mission aggregates available",
            "config family available",
        ],
        "value": [
            len(launch_df), int(launch_df["launch_time_utc"].isna().sum()),
            int(launch_df["Location"].isna().sum()), int(launch_df["Location"].nunique()),
            int(launch_df["Comb Launch Site"].nunique()), int(launch_df["payload_count"].notna().sum()),
            int(launch_df["rocket_family"].notna().sum()),
        ],
    }
)
prep_summary
"""
    ),
    md("## 3. U.S. Scope and Facility Normalization"),
    code(
        """
def infer_us_facility_group(location: str) -> str:
    location = "" if pd.isna(location) else str(location)
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


us_launches = launch_df.loc[launch_df["Country_Code"] == "US"].copy()
us_launches["facility_group"] = us_launches["Location"].apply(infer_us_facility_group)

us_summary = pd.DataFrame(
    {
        "metric": [
            "U.S. launches", "U.S. min launch date", "U.S. max launch date",
            "unique U.S. raw location strings", "unique U.S. combined launch sites",
        ],
        "value": [
            len(us_launches), str(us_launches["launch_date"].min()), str(us_launches["launch_date"].max()),
            int(us_launches["Location"].nunique()), int(us_launches["Comb Launch Site"].nunique()),
        ],
    }
)
us_summary
"""
    ),
    md("## 4. Launch-Weather Merge and Engineered Features"),
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
    weather["short_duration_precip_max"] = (
        weather[available_short_duration_cols].max(axis=1, skipna=True)
        if available_short_duration_cols else float("nan")
    )
    return weather


def add_prelaunch_history(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    valid = df[group_col].notna()
    grp = df.loc[valid].groupby(group_col, sort=False)
    df.loc[valid, f"{prefix}_prior_launches"] = grp.cumcount()
    df.loc[valid, f"{prefix}_prior_successes"] = grp["launch_success_binary"].cumsum() - df.loc[valid, "launch_success_binary"]
    df[f"{prefix}_success_rate_prelaunch"] = df[f"{prefix}_prior_successes"] / df[f"{prefix}_prior_launches"].replace(0, np.nan)
    first_time = grp["launch_time_utc"].transform("min")
    prev_time = grp["launch_time_utc"].shift(1)
    df.loc[valid, f"{prefix}_years_since_first_launch"] = (
        (df.loc[valid, "launch_time_utc"] - first_time).dt.total_seconds() / (365.25 * 24 * 3600)
    )
    df.loc[valid, f"days_since_previous_launch_{prefix}"] = (
        (df.loc[valid, "launch_time_utc"] - prev_time).dt.total_seconds() / (24 * 3600)
    )
    return df


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

us_launch_weather = pd.concat(weather_merges, ignore_index=True).sort_values("launch_time_utc").reset_index(drop=True)
us_launch_weather["launch_outcome_group"] = us_launch_weather["Launch Status"].where(
    us_launch_weather["Launch Status"] == "Success", "Not Success"
)
us_launch_weather["launch_success_binary"] = (us_launch_weather["Launch Status"] == "Success").astype(int)
us_launch_weather["launch_failure_binary"] = (us_launch_weather["Launch Status"] != "Success").astype(int)
us_launch_weather["is_partial_failure"] = (us_launch_weather["Launch Status"] == "Partial Failure").astype(int)
us_launch_weather["is_failure"] = (us_launch_weather["Launch Status"] == "Failure").astype(int)
us_launch_weather["is_prelaunch_failure"] = (us_launch_weather["Launch Status"] == "Prelaunch Failure").astype(int)
us_launch_weather["rocket_payload_leo"] = pd.to_numeric(us_launch_weather["Rocket Payload to LEO"], errors="coerce")
us_launch_weather["rocket_price_musd"] = pd.to_numeric(us_launch_weather["Rocket Price"], errors="coerce")
us_launch_weather["rocket_price_adjusted_musd"] = pd.to_numeric(us_launch_weather["Rocket Price CPI Adjusted"], errors="coerce")
us_launch_weather["usd_per_kg_leo_adjusted"] = pd.to_numeric(us_launch_weather["USD/kg to LEO CPI Adjusted"], errors="coerce")
top_orgs = us_launch_weather["Rocket Organisation"].value_counts().head(8).index
us_launch_weather["rocket_org_grouped"] = us_launch_weather["Rocket Organisation"].where(
    us_launch_weather["Rocket Organisation"].isin(top_orgs), "Other"
)
us_launch_weather["payload_bin"] = pd.cut(us_launch_weather["rocket_payload_leo"], bins=[0, 500, 2000, 10000, 50000, 500000], include_lowest=True)
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
us_launch_weather["launch_season"] = us_launch_weather["launch_month"].map(season_map)
us_launch_weather["launch_hour_local"] = us_launch_weather["launch_time_lstd"].dt.hour
us_launch_weather["launch_hour_bin"] = pd.cut(us_launch_weather["launch_hour_local"], bins=[-1, 5, 11, 17, 23], labels=["Overnight", "Morning", "Afternoon", "Evening"])
us_launch_weather["era_group"] = pd.cut(
    us_launch_weather["launch_year"],
    bins=[1950, 1979, 1999, 2025],
    labels=["Early space age (1951-1979)", "Transition era (1980-1999)", "Modern era (2000-2024)"],
    include_lowest=True,
)
us_launch_weather["precip_positive_flag"] = us_launch_weather["HourlyPrecipitation"].fillna(0).gt(0)
us_launch_weather["weather_type_reported_flag"] = us_launch_weather["HourlyPresentWeatherType"].fillna("").astype(str).str.len().gt(0)
us_launch_weather["high_wind_flag"] = us_launch_weather["HourlyWindSpeed"].ge(15)
us_launch_weather["low_visibility_flag"] = us_launch_weather["HourlyVisibility"].le(5)
us_launch_weather["high_humidity_flag"] = us_launch_weather["HourlyRelativeHumidity"].ge(80)
us_launch_weather["dewpoint_depression"] = us_launch_weather["HourlyDryBulbTemperature"] - us_launch_weather["HourlyDewPointTemperature"]
us_launch_weather["has_any_precip_signal"] = (
    us_launch_weather[["precip_positive_flag", "present_weather_rain_flag"]]
    .fillna(False).astype(bool).any(axis=1)
)
us_launch_weather["high_wind_and_low_visibility_flag"] = us_launch_weather["high_wind_flag"].fillna(False).astype(bool) & us_launch_weather["low_visibility_flag"].fillna(False).astype(bool)
us_launch_weather["high_wind_and_high_humidity_flag"] = us_launch_weather["high_wind_flag"].fillna(False).astype(bool) & us_launch_weather["high_humidity_flag"].fillna(False).astype(bool)
us_launch_weather["rain_and_low_visibility_flag"] = us_launch_weather["has_any_precip_signal"].fillna(False).astype(bool) & us_launch_weather["low_visibility_flag"].fillna(False).astype(bool)
us_launch_weather["wind_x_visibility"] = us_launch_weather["HourlyWindSpeed"] * us_launch_weather["HourlyVisibility"]
us_launch_weather["wind_x_humidity"] = us_launch_weather["HourlyWindSpeed"] * us_launch_weather["HourlyRelativeHumidity"]
us_launch_weather["weather_match_quality_bin"] = pd.cut(us_launch_weather["weather_time_diff_minutes"], bins=[0, 15, 30, 60, 120], labels=["0-15 min", "15-30 min", "30-60 min", "60-120 min"], include_lowest=True)
us_launch_weather["multi_payload_flag"] = us_launch_weather["payload_count"].fillna(0).gt(1)
us_launch_weather["rocket_price_adjusted_missing_flag"] = us_launch_weather["rocket_price_adjusted_musd"].isna()
us_launch_weather["usd_per_kg_leo_adjusted_missing_flag"] = us_launch_weather["usd_per_kg_leo_adjusted"].isna()
us_launch_weather["config_fairing_diameter_missing_flag"] = us_launch_weather["config_fairing_diameter"].isna()
us_launch_weather["config_fairing_height_missing_flag"] = us_launch_weather["config_fairing_height"].isna()
us_launch_weather["HourlyVisibility_missing_flag"] = us_launch_weather["HourlyVisibility"].isna()
us_launch_weather["HourlyAltimeterSetting_missing_flag"] = us_launch_weather["HourlyAltimeterSetting"].isna()
us_launch_weather["HourlyWetBulbTemperature_missing_flag"] = us_launch_weather["HourlyWetBulbTemperature"].isna()

for group_col, prefix in [("rocket_family", "family"), ("Rocket Organisation", "org"), ("Rocket Name", "config"), ("facility_group", "site")]:
    us_launch_weather = add_prelaunch_history(us_launch_weather, group_col, prefix)

us_launch_weather["family_launches_so_far"] = us_launch_weather["family_prior_launches"]
us_launch_weather["site_launches_so_far"] = us_launch_weather["site_prior_launches"]
us_launch_weather["config_launches_so_far"] = us_launch_weather["config_prior_launches"]
us_launch_weather["org_launches_so_far"] = us_launch_weather["org_prior_launches"]

for raw_col, new_col in [
    ("HourlyWindSpeed", "site_wind_speed_z"),
    ("HourlyVisibility", "site_visibility_z"),
    ("HourlyRelativeHumidity", "site_humidity_z"),
    ("dewpoint_depression", "site_dewpoint_depression_z"),
    ("HourlyAltimeterSetting", "site_altimeter_setting_z"),
]:
    site_mean = us_launch_weather.groupby("facility_group")[raw_col].transform("mean")
    site_std = us_launch_weather.groupby("facility_group")[raw_col].transform("std")
    us_launch_weather[new_col] = (us_launch_weather[raw_col] - site_mean) / site_std.replace(0, np.nan)

weather_merge_coverage = pd.DataFrame(weather_coverage_rows).sort_values("matched_launches", ascending=False).reset_index(drop=True)
feature_engineering_summary = pd.DataFrame(
    {
        "feature_family": [
            "prelaunch historical rates", "launch-history counts", "cadence features",
            "site-relative weather z-scores", "weather interactions", "missingness indicators",
        ],
        "examples": [
            "family_success_rate_prelaunch, org_success_rate_prelaunch, config_success_rate_prelaunch",
            "family_launches_so_far, site_launches_so_far, config_launches_so_far",
            "days_since_previous_launch_site, days_since_previous_launch_family",
            "site_wind_speed_z, site_visibility_z, site_humidity_z",
            "high_wind_and_low_visibility_flag, high_wind_and_high_humidity_flag, rain_and_low_visibility_flag",
            "rocket_price_adjusted_missing_flag, HourlyVisibility_missing_flag, HourlyAltimeterSetting_missing_flag",
        ],
    }
)
feature_engineering_summary
"""
    ),
    md("## 5. Prepared Output Audit"),
    code(
        """
matched_launch_weather = us_launch_weather.loc[us_launch_weather["weather_matched"]].copy()
weather_feature_availability = pd.DataFrame(
    {
        "feature": WEATHER_NUMERIC_COLUMNS + WEATHER_TEXT_COLUMNS + [
            "present_weather_rain_flag", "present_weather_fog_flag",
            "present_weather_thunder_flag", "cloud_cover_broken_or_overcast_flag", "short_duration_precip_max",
        ],
        "non_null_share": [
            matched_launch_weather[col].notna().mean() if col in matched_launch_weather.columns else 0
            for col in WEATHER_NUMERIC_COLUMNS + WEATHER_TEXT_COLUMNS + [
                "present_weather_rain_flag", "present_weather_fog_flag",
                "present_weather_thunder_flag", "cloud_cover_broken_or_overcast_flag", "short_duration_precip_max",
            ]
        ],
    }
).sort_values("non_null_share", ascending=False)
weather_feature_availability
"""
    ),
    md("## 6. Write Prepared Outputs"),
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

facility_feature_availability_export = (
    matched_launch_weather.groupby("facility_group")[
        [c for c in WEATHER_NUMERIC_COLUMNS if c in matched_launch_weather.columns]
    ]
    .agg(lambda s: s.notna().mean())
    .T
    .reset_index()
    .rename(columns={"index": "feature"})
)

us_site_summary_export.to_csv(OUTPUT_DIR / "us_launch_site_summary.csv", index=False)
weather_merge_coverage.to_csv(OUTPUT_DIR / "weather_merge_coverage.csv", index=False)
weather_feature_availability.to_csv(OUTPUT_DIR / "weather_feature_availability.csv", index=False)
facility_feature_availability_export.to_csv(OUTPUT_DIR / "facility_weather_feature_availability.csv", index=False)
us_launch_weather.to_csv(OUTPUT_DIR / "us_launch_weather_merged.csv", index=False)

pd.DataFrame(
    {
        "file": [
            str(OUTPUT_DIR / "us_launch_site_summary.csv"),
            str(OUTPUT_DIR / "weather_merge_coverage.csv"),
            str(OUTPUT_DIR / "weather_feature_availability.csv"),
            str(OUTPUT_DIR / "facility_weather_feature_availability.csv"),
            str(OUTPUT_DIR / "us_launch_weather_merged.csv"),
        ]
    }
)
"""
    ),
]

eda_cells = [
    md(
        """
# Rocket Launch EDA for Success/Failure Modeling

This notebook is analysis-only. It assumes `data_prep.ipynb` has already been run and that the prepared launch-weather event table exists in `data/derived/us_launch_weather_merged.csv`.

The goal of the EDA is feature discovery for downstream modeling. That means the emphasis is not just on descriptive plots, but on identifying variables that are interpretable, reasonably available, and still informative after the major confounders of **site** and **era** are considered.
"""
    ),
    code(analysis_header),
    md("## 1. Prepared Sample Overview"),
    code(
        """
analysis_input_summary = pd.DataFrame(
    {
        "metric": [
            "prepared U.S. launches",
            "weather-matched launches",
            "weather match rate",
            "success share",
            "failure share",
            "partial-failure share",
            "date_from",
            "date_to",
            "facility groups",
        ],
        "value": [
            len(us_launch_weather),
            int(us_launch_weather["weather_matched"].sum()),
            us_launch_weather["weather_matched"].mean(),
            us_launch_weather["launch_success_binary"].mean(),
            us_launch_weather["is_failure"].mean(),
            us_launch_weather["is_partial_failure"].mean(),
            str(us_launch_weather["launch_date"].min()),
            str(us_launch_weather["launch_date"].max()),
            int(us_launch_weather["facility_group"].nunique()),
        ],
    }
)
analysis_input_summary
"""
    ),
    code(
        """
us_status_counts = (
    us_launch_weather["Launch Status"].value_counts().rename_axis("launch_status").reset_index(name="launches")
)
us_by_decade = (
    us_launch_weather.groupby("launch_decade").size().rename("launches").reset_index().sort_values("launch_decade")
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.barplot(data=us_status_counts, y="launch_status", x="launches", ax=axes[0], color="#4c78a8")
axes[0].set_title("Outcome Counts")
axes[0].set_xlabel("Launches")
axes[0].set_ylabel("")
sns.barplot(data=us_by_decade, x="launch_decade", y="launches", ax=axes[1], color="#72b7b2")
axes[1].set_title("Launches by Decade")
axes[1].set_xlabel("Launch decade")
axes[1].set_ylabel("Launches")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### Interpretation

The class imbalance remains substantial, so every apparent predictor should be read in that context. The decade plot also reinforces a core lesson from the earlier EDA: launch outcomes are deeply tied to historical period, which is why later sections repeatedly compare pooled signal against site/era-adjusted signal.
"""
    ),
    md("## 2. Feature Availability and Weather Match Quality"),
    code(
        """
plt.figure(figsize=(10, 5))
sns.barplot(data=weather_merge_coverage.sort_values("match_rate", ascending=False), x="match_rate", y="facility_group", color="#59a14f")
plt.title("Weather Match Coverage by Facility")
plt.xlabel("Matched share within 2 hours")
plt.ylabel("")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

weather_feature_availability
"""
    ),
    md(
        """
#### Interpretation

The match-quality and availability checks are still the first gate for weather features. A variable can only become a stable modeling feature if it has enough usable coverage at the sites that drive most of the sample. This is why wind remains more attractive than many other weather fields even before formal screening starts.
"""
    ),
    md("## 3. Historical Reliability and Launch-Maturity Features"),
    code(
        """
history_feature_profiles = []
for feature in [
    "family_success_rate_prelaunch",
    "org_success_rate_prelaunch",
    "config_success_rate_prelaunch",
    "family_launches_so_far",
    "site_launches_so_far",
    "days_since_previous_launch_site",
    "days_since_previous_launch_family",
]:
    profile = (
        us_launch_weather.groupby("launch_outcome_group")[feature]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    profile["feature"] = feature
    history_feature_profiles.append(profile)
history_feature_profiles = pd.concat(history_feature_profiles, ignore_index=True)
history_feature_profiles
"""
    ),
    code(
        """
history_feature_diff_rows = []
history_visual_features = [
    "family_success_rate_prelaunch",
    "org_success_rate_prelaunch",
    "config_success_rate_prelaunch",
    "family_launches_so_far",
    "site_launches_so_far",
    "days_since_previous_launch_site",
    "days_since_previous_launch_family",
]

for feature in history_visual_features:
    feature_df = us_launch_weather[[feature, "launch_success_binary"]].dropna().copy()
    if feature_df.empty or feature_df["launch_success_binary"].nunique() < 2:
        continue

    summary = (
        feature_df.groupby("launch_success_binary")[feature]
        .agg(["mean", "median", "count"])
        .reindex([0, 1])
    )
    if summary[["mean", "median"]].isna().any().any():
        continue

    history_feature_diff_rows.append(
        {
            "feature": feature,
            "failure_mean": summary.loc[0, "mean"],
            "success_mean": summary.loc[1, "mean"],
            "mean_gap_success_minus_failure": summary.loc[1, "mean"] - summary.loc[0, "mean"],
            "failure_median": summary.loc[0, "median"],
            "success_median": summary.loc[1, "median"],
            "median_gap_success_minus_failure": summary.loc[1, "median"] - summary.loc[0, "median"],
            "paired_sample_size": int(summary["count"].sum()),
        }
    )

history_feature_differences = pd.DataFrame(history_feature_diff_rows).sort_values(
    "mean_gap_success_minus_failure", key=lambda s: s.abs(), ascending=True
)
history_feature_differences
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

mean_plot = history_feature_differences.sort_values(
    "mean_gap_success_minus_failure", key=lambda s: s.abs(), ascending=True
)
median_plot = history_feature_differences.sort_values(
    "median_gap_success_minus_failure", key=lambda s: s.abs(), ascending=True
)

sns.barplot(
    data=mean_plot,
    x="mean_gap_success_minus_failure",
    y="feature",
    ax=axes[0],
    color="#4e79a7",
)
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title("Mean Difference: Success - Failure")
axes[0].set_xlabel("Gap in mean value")
axes[0].set_ylabel("")

sns.barplot(
    data=median_plot,
    x="median_gap_success_minus_failure",
    y="feature",
    ax=axes[1],
    color="#f28e2b",
)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Median Difference: Success - Failure")
axes[1].set_xlabel("Gap in median value")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
history_specs = [
    ("family_success_rate_prelaunch", "Pre-Launch Family Success Rate Quantiles"),
    ("org_success_rate_prelaunch", "Pre-Launch Org Success Rate Quantiles"),
    ("config_success_rate_prelaunch", "Pre-Launch Config Success Rate Quantiles"),
    ("family_launches_so_far", "Prior Family Launch Count Quantiles"),
]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (feature, title) in zip(axes.flat, history_specs):
    profile = binned_success_profile(us_launch_weather, feature, quantiles=5, min_launches=10)
    sns.barplot(data=profile, x="feature_bin", y="launches", ax=ax, color="#eeca3b")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Launch count")
    ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    sns.pointplot(data=profile, x="feature_bin", y="success_rate", ax=ax2, color="#003f5c")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### Interpretation

These history features are among the most promising additions because they are directly aligned with reliability learning and are much closer to what a model would actually want to know: how mature is this family, configuration, organisation, or site before the current launch happens? The time-safe prelaunch versions are especially important because they avoid the leakage risk in the original `family_success_rate_pct`.

The mean and median gap charts help distinguish broad shifts from outlier-driven shifts. When both plots point in the same direction for a feature, the signal is less likely to be coming from a small number of extreme launches and more likely to reflect a stable difference between successful and unsuccessful launches.
"""
    ),
    md("## 4. Vehicle, Mission, and Capability Features"),
    code(
        """
launch_profile_specs = [
    ("mission_mass", "Mission Mass Quantiles"),
    ("rocket_payload_leo", "Rocket Payload to LEO Quantiles"),
    ("config_liftoff_thrust", "Liftoff Thrust Quantiles"),
    ("config_rocket_height", "Rocket Height Quantiles"),
]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (feature, title) in zip(axes.flat, launch_profile_specs):
    profile = binned_success_profile(us_launch_weather, feature, quantiles=5, min_launches=10)
    sns.barplot(data=profile, x="feature_bin", y="launches", ax=ax, color="#cdb4db")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Launch count")
    ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    sns.pointplot(data=profile, x="feature_bin", y="success_rate", ax=ax2, color="#1d3557")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### Interpretation

The launch-side capability features still matter because they partition the launch population into more comparable vehicle classes. The most useful ones are the variables that keep separating outcomes even after site and era are controlled for later in the notebook, which is why height, thrust, and payload proxies remain central candidates.
"""
    ),
    md("## 5. Site-Relative Weather Features"),
    code(
        """
site_relative_weather_profiles = []
for feature in ["site_wind_speed_z", "site_visibility_z", "site_humidity_z", "site_dewpoint_depression_z"]:
    profile = (
        matched_launch_weather.groupby("launch_outcome_group")[feature]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    profile["feature"] = feature
    site_relative_weather_profiles.append(profile)
site_relative_weather_profiles = pd.concat(site_relative_weather_profiles, ignore_index=True)
site_relative_weather_profiles
"""
    ),
    code(
        """
plt.figure(figsize=(12, 6))
site_relative_plot = matched_launch_weather.melt(
    id_vars=["launch_outcome_group"],
    value_vars=["site_wind_speed_z", "site_visibility_z", "site_humidity_z", "site_dewpoint_depression_z"],
    var_name="feature",
    value_name="value",
)
sns.boxplot(data=site_relative_plot, x="feature", y="value", hue="launch_outcome_group")
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.title("Site-Relative Weather Features by Outcome")
plt.xlabel("Feature")
plt.ylabel("Site-level z-score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### Interpretation

The site-relative weather features are important because they answer a better question than raw weather: was the launch happening in weather that was unusual for that facility? This is especially useful for generalizable modeling because a wind level that is routine at one site may be operationally unusual at another.
"""
    ),
    md("## 6. Weather Interaction Features"),
    code(
        """
weather_interaction_summary = []
for feature in [
    "high_wind_and_low_visibility_flag",
    "high_wind_and_high_humidity_flag",
    "rain_and_low_visibility_flag",
    "high_wind_flag",
    "low_visibility_flag",
    "has_any_precip_signal",
]:
    flag_df = matched_launch_weather[[feature, "launch_success_binary", "Launch Id"]].copy()
    flag_mask = flag_df[feature].fillna(False).astype(bool)
    weather_interaction_summary.append(
        {
            "feature": feature,
            "launches_with_flag": int(flag_mask.sum()),
            "success_rate_flag_true": flag_df.loc[flag_mask, "launch_success_binary"].mean(),
            "success_rate_flag_false": flag_df.loc[~flag_mask, "launch_success_binary"].mean(),
            "success_rate_gap_true_minus_false": (
                flag_df.loc[flag_mask, "launch_success_binary"].mean()
                - flag_df.loc[~flag_mask, "launch_success_binary"].mean()
            ),
        }
    )
weather_interaction_summary = pd.DataFrame(weather_interaction_summary).sort_values(
    "success_rate_gap_true_minus_false"
)
plt.figure(figsize=(10, 5))
sns.barplot(data=weather_interaction_summary, x="success_rate_gap_true_minus_false", y="feature", color="#d55e00")
plt.axvline(0, color="black", linewidth=1)
plt.title("Weather Interaction and Flag Success-Rate Gaps")
plt.xlabel("Success rate with flag - success rate without flag")
plt.ylabel("")
plt.tight_layout()
plt.show()
weather_interaction_summary
"""
    ),
    md(
        """
#### Interpretation

These interaction flags matter because launch-weather risk is unlikely to be additive in a simple way. Combined poor conditions are often more operationally meaningful than any single raw field. If an interaction remains useful after site/era adjustment in the screening section, it becomes a strong candidate for tree models or manually engineered linear features.
"""
    ),
    md("## 7. Missingness Indicators"),
    code(missingness_code),
    code(
        """
plt.figure(figsize=(10, 5))
sns.barplot(
    data=missingness_signal_summary,
    x="missing_rate_gap_not_success_minus_success",
    y="feature",
    hue="feature_group",
)
plt.axvline(0, color="black", linewidth=1)
plt.title("Difference in Missingness Between Not-Success and Success Samples")
plt.xlabel("Missing rate in Not Success - missing rate in Success")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### Interpretation

Missingness indicators are worth carrying forward because the EDA suggests some fields are missing in outcome-skewed ways. That can reflect documentation quality, era, site, or program maturity. For linear models, this often argues for explicit missing flags. For tree models, it suggests that sparse variables should not be judged only by their observed values.
"""
    ),
    md("## 8. Modeling-Oriented Feature Screening"),
    code(feature_screen_code),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
coverage_signal_plot = feature_screening_summary.dropna(subset=["abs_pooled_corr_with_success"]).copy()
sns.scatterplot(
    data=coverage_signal_plot,
    x="non_null_share",
    y="abs_pooled_corr_with_success",
    hue="feature_group",
    s=110,
    ax=axes[0],
)
for _, row in coverage_signal_plot.nlargest(10, "abs_pooled_corr_with_success").iterrows():
    axes[0].text(row["non_null_share"] + 0.01, row["abs_pooled_corr_with_success"], row["feature"], fontsize=8)
axes[0].set_title("Coverage vs. Pooled Signal")
axes[0].set_xlabel("Non-null share")
axes[0].set_ylabel("Absolute correlation with success")
axes[0].set_xlim(0, 1.05)

adjusted_signal_plot = feature_screening_summary.dropna(subset=["abs_within_site_era_z_gap"]).copy()
sns.scatterplot(
    data=adjusted_signal_plot,
    x="non_null_share",
    y="abs_within_site_era_z_gap",
    hue="feature_group",
    s=110,
    ax=axes[1],
    legend=False,
)
for _, row in adjusted_signal_plot.nlargest(10, "abs_within_site_era_z_gap").iterrows():
    axes[1].text(row["non_null_share"] + 0.01, row["abs_within_site_era_z_gap"], row["feature"], fontsize=8)
axes[1].set_title("Coverage vs. Site/Era-Adjusted Signal")
axes[1].set_xlabel("Non-null share")
axes[1].set_ylabel("Absolute within-site/era z-gap")
axes[1].set_xlim(0, 1.05)
plt.tight_layout()
plt.show()

feature_screening_summary
"""
    ),
    md(
        """
#### Interpretation

This screening view is the main bridge from EDA to modeling. Variables that look strong only in pooled comparisons may just be reflecting historical composition. The most attractive candidates are the ones that keep some signal even after site and era are taken into account, while still retaining enough coverage to be practical.
"""
    ),
    md("## 9. Candidate Feature Tiers"),
    code(model_tiers_code),
    code(
        """
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(
    data=model_feature_tiers,
    y="modeling_tier",
    hue="feature_group",
    order=["core modeling candidate", "secondary candidate", "exploratory / sparse candidate"],
    ax=ax,
)
ax.set_title("Candidate Feature Tiers for Modeling")
ax.set_xlabel("Feature count")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

model_feature_tiers
"""
    ),
    md(
        """
#### Interpretation

The feature tiers are not a final model specification, but they are a useful starting point. The **core** tier is the best first-pass feature set for baseline logistic or regularized models because it balances signal and coverage. The **secondary** tier is the natural expansion set for richer tree-based models or ablation studies. The **exploratory** tier is best treated as optional until it proves stable.
"""
    ),
    md("### Candidate Variable Definitions and EDA-Based Justification"),
    md(candidate_table_md),
    md(
        """
#### Interpretation

The strongest additions are the **time-safe historical reliability features** and the **site-relative weather features**. Together, they push the notebook much closer to a model-design document: they are interpretable, aligned with what could be known at prediction time, and supported by the EDA rather than added only on intuition.
"""
    ),
    md("## 10. Write Derived Analysis Tables"),
    code(
        """
feature_screening_summary.to_csv(OUTPUT_DIR / "feature_screening_summary.csv", index=False)
missingness_signal_summary.to_csv(OUTPUT_DIR / "missingness_signal_summary.csv", index=False)
model_feature_tiers.to_csv(OUTPUT_DIR / "model_feature_tiers.csv", index=False)

pd.DataFrame(
    {
        "file": [
            str(OUTPUT_DIR / "feature_screening_summary.csv"),
            str(OUTPUT_DIR / "missingness_signal_summary.csv"),
            str(OUTPUT_DIR / "model_feature_tiers.csv"),
        ]
    }
)
"""
    ),
    md(
        """
## 11. Bottom Line

The new feature engineering moves the project in a more modeling-ready direction. The most important improvements are:
- time-safe prelaunch history features that replace leakage-prone full-history summaries
- site-relative weather features that make cross-site weather comparisons more meaningful
- explicit weather interaction flags and missingness indicators
- cadence and maturity features that help distinguish event-level risk from simple historical composition

At this point, the notebook should support building a sensible first-pass feature set for both linear and tree-based models without relying only on raw launch or raw weather variables.
"""
    ),
]

nb_meta = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.14.0",
    },
}

(ROOT / "data_prep.ipynb").write_text(
    json.dumps({"cells": data_prep_cells, "metadata": nb_meta, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
(ROOT / "EDA.ipynb").write_text(
    json.dumps({"cells": eda_cells, "metadata": nb_meta, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1),
    encoding="utf-8",
)

print("Rebuilt data_prep.ipynb and EDA.ipynb")
