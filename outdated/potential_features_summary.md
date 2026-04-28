# Potential Features Summary

This document summarizes additional candidate features for the rocket launch success/failure project beyond the first-pass weather and launch EDA already added to `EDA.ipynb`.

## Current Scope

The notebook currently uses a smaller initial set of launch-side and weather-side features because they are relatively interpretable and useful for first-pass EDA.

That is not the full feature space available in the data.

## Additional Launch Dataset Features

From `Launches.csv`, additional candidate features include:

- `Rocket Name`
- `Rocket Organisation`
- `Location`
- `facility_group`
- `Launch Year`
- `Launch Year Mon`
- `Rocket Price`
- `Rocket Price CPI Adjusted`
- `Rocket Payload to LEO`
- `USD/kg to LEO`
- `USD/kg to LEO CPI Adjusted`
- `Launch Suborbital`

From `Missions.csv`, after joining on `Launch Id`, candidate features include:

- payload count
- total mission mass

From `Families.csv` and `Configs.csv`, after joining in rocket family/configuration information, candidate features include:

- rocket family
- stages
- thrust
- fairing size
- configuration status
- strap-ons
- height

## Additional LCD Weather Features

From the LCD weather files, additional candidate features include:

- `HourlyAltimeterSetting`
- `HourlyWetBulbTemperature`
- `HourlyPressureChange`
- `HourlyPressureTendency`
- `HourlyPresentWeatherType`
- `HourlySkyConditions`
- `DailyPeakWindSpeed`
- `DailyPeakWindDirection`
- `DailyPrecipitation`
- `DailyAverageWindSpeed`
- `DailyMaximumDryBulbTemperature`
- `DailyMinimumDryBulbTemperature`

Short-duration precipitation fields are also available for some sites:

- `ShortDurationPrecipitationValue005`
- `ShortDurationPrecipitationValue010`
- `ShortDurationPrecipitationValue015`
- `ShortDurationPrecipitationValue030`
- and the related short-duration precipitation fields at other intervals

## Engineered Features

Useful engineered features that could be created from the merged launch and weather data include:

- time to nearest weather observation
- launch hour in local time
- season
- month
- wet/dry flag
- high-wind flag
- low-visibility flag
- dew point depression: `temperature - dew point`
- pressure anomaly within facility
- rolling precipitation in prior hours
- whether present weather indicates fog, rain, or thunder
- payload class
- operator fixed effects
- facility fixed effects
- pre-1980, post-1980, or other era indicators

## Strong Next Candidates

The strongest next candidates to add are:

1. `payload_count` and `mission_mass` from `Missions.csv`
2. `stages`, `strap-ons`, and `liftoff thrust` from `Configs.csv`
3. `HourlyAltimeterSetting` and `HourlyWetBulbTemperature` from LCD
4. parsed weather-condition flags from `HourlyPresentWeatherType`
5. parsed cloud-cover flags from `HourlySkyConditions`
6. short-duration precipitation fields where available

## Implemented Strong Candidates

The following strong next candidates have now been implemented in the notebook and merged dataset:

- `payload_count` from `Missions.csv`
- `mission_mass` from `Missions.csv`
- `mission_rows` from `Missions.csv`
- `rocket_family` from `Families.csv` via `Configs.csv`
- `config_status`
- `config_liftoff_thrust`
- `config_payload_leo`
- `config_payload_gto`
- `config_stages`
- `config_strap_ons`
- `config_rocket_height`
- `config_fairing_diameter`
- `config_fairing_height`
- `family_success_rate_pct`
- `HourlyAltimeterSetting`
- `HourlyWetBulbTemperature`
- parsed rain/fog/thunder flags from `HourlyPresentWeatherType`
- parsed broken/overcast cloud flag from `HourlySkyConditions`
- `short_duration_precip_max`, built from the available short-duration precipitation fields

These implemented features are now available in the merged launch-weather table and in the updated EDA notebook.

## Notes on Implemented Features

- `Missions.csv` joins by `Launch Id`.
- `Configs.csv` joins cleanly by exact `Rocket Name == Config` in this dataset.
- Some config-style fields have much better coverage than others. For example, stages and family coverage are strong, while fairing dimensions are more limited.
- `short_duration_precip_max` is implemented, but it is extremely sparse in the current LCD files and should be treated as an exploratory feature rather than a core modeling variable.
- Parsed weather and cloud flags are meant to make `HourlyPresentWeatherType` and `HourlySkyConditions` easier to use in EDA and later modeling.

## Lower-Priority Features

These are less attractive for the next pass:

- monthly normals or monthly summary fields from LCD
- very sparse features like `HourlyWindGustSpeed` unless coverage is acceptable
- highly site-specific fields that only exist in one or two weather files

## Recommended Next Step

A useful next pass would be to:

1. audit feature availability and missingness across all implemented variables
2. select a smaller modeling-ready feature subset with acceptable coverage
3. decide how to handle sparse but potentially informative variables such as short-duration precipitation and fairing geometry
