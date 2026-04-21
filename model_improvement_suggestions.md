# Model Improvement Suggestions

The strongest path to improving these models is probably **not** switching to a more complex algorithm. The current results suggest that performance is being constrained more by **feature quality, missingness, temporal heterogeneity, and site/era differences** than by model class.

Across the current notebooks and derived summary files, the most consistent pattern is that the strongest predictors are **historical reliability and launch maturity features**, including:

- `family_success_rate_prelaunch`
- `org_success_rate_prelaunch`
- `config_success_rate_prelaunch`
- `site_launches_so_far`
- `family_launches_so_far`

In the modeling notebook, **logistic regression** matched or slightly beat the nonlinear models on the held-out test split. That usually means the biggest gains are more likely to come from **better formulation and better features** than from a more flexible learner.

## Highest-Value Improvements

### 1. Use time-aware validation instead of random splits

The data spans very different launch eras, and the subgroup analysis in the modeling notebook already shows weaker model behavior in the **1980-1999 transition era**. A random stratified split mixes eras together and can let the model benefit from future-era patterns that may not generalize.

The better approach is to:

- train on earlier launches
- validate on a later block
- test on the latest block

This will produce a more realistic estimate of how well the model generalizes over time.

### 2. Model site and era heterogeneity explicitly

The facility-level and era-level diagnostics show that the launch process is not homogeneous. Weather effects vary by site, and model performance varies by era.

Likely better options:

- fit separate models by era
- fit separate models by major launch site
- use a global model with site and era interaction terms

Right now, one model is trying to describe early-space-age launches and modern launches with the same decision surface.

### 3. Improve weather coverage before adding more weather complexity

Weather features do appear to help, especially wind-related variables, but several weather fields have substantial missingness or outcome-skewed coverage:

- `HourlyVisibility`
- `HourlyAltimeterSetting`
- `HourlyWetBulbTemperature`

That means adding more weather variables will only help if the coverage and matching quality improve.

Better next steps:

- audit the launch-to-weather matching process more carefully
- add lagged or rolling weather windows such as recent max wind or recent precipitation
- expand site-relative anomaly features instead of relying only on raw weather values
- avoid very sparse weather fields unless they can be made materially more complete

### 4. Expand prelaunch reliability features in a leakage-safe way

These are already the strongest features in the project, so expanding them is one of the most promising directions.

Potential additions:

- rolling recent success rates instead of only cumulative rates
- success rates over the last 5 or 10 launches
- launches in the prior 30, 90, or 365 days
- recent failure indicators for family, organization, configuration, or site
- years since first launch for family, organization, configuration, or site

These features must be built strictly from information available **before** the launch date.

### 5. Add targeted interaction features

Because logistic regression already performs well, carefully chosen interaction terms may improve performance more efficiently than switching to a more complex model.

Most plausible interactions:

- reliability x era
- reliability x facility
- wind x facility
- mission mass or payload x family/config maturity
- days-since-previous-launch x maturity or cadence

The current results suggest that weather and configuration effects are context dependent rather than globally constant.

### 6. Optimize more directly for the project objective

The best current logistic model still has only moderate failure recall on the test split. If the real goal is to identify risky launches, the modeling objective may need to shift away from balanced accuracy alone.

Better alternatives:

- optimize threshold for higher minority-class recall
- tune directly against PR AUC
- treat the problem as a ranking problem and focus on top-risk launches

If this is a screening model, catching more failures may matter more than maximizing a single average metric.

### 7. Calibrate final predicted probabilities

Calibration plots were added to the notebook, which is useful. The next step would be explicit calibration of the final model with:

- Platt scaling
- isotonic regression

This matters if the model will be interpreted as a true probability model instead of only a ranking model.

### 8. Replace some hard categorical effects with structural summaries

The logistic regression coefficients indicate that grouped operator and facility categories carry a lot of weight. That helps fit the current sample, but it may reduce portability and make the model too dependent on memorized identities.

Possible replacements:

- operator prior launches
- operator prior failures
- operator recent success rate
- facility maturity
- facility recent cadence
- family-level historical volatility

This would make the model less dependent on raw labels and more dependent on interpretable operational structure.

## Recommended Next Three Steps

If only three improvements are implemented next, the best priorities are likely:

1. Rebuild evaluation using chronological train/validation/test splits.
2. Engineer richer leakage-safe reliability and cadence features.
3. Fit either site-specific or era-specific logistic models, or add targeted site/era interaction terms to a global logistic model.

## Why This Is The Best Direction

The current evidence does **not** suggest that the main limitation is lack of model flexibility. Instead, it suggests:

- the strongest signal already comes from structured historical features
- weather signal exists but is limited by missingness and uneven coverage
- launch behavior changes across facilities and eras
- the best current model is still the simpler, more interpretable logistic regression

That combination usually means the next improvement should come from **better data design and better feature engineering**, not just a more complex learner.

## Update Note: April 19, 2026

Two follow-up V2 notebooks were added after the original write-up:

- `modeling_V2.ipynb`
- `modeling_weather_V2.ipynb`

These provide new evidence about which directions are genuinely promising and which ones are more fragile.

### 1. `modeling_V2.ipynb`: Rolling History And Regularized Logistic

This notebook tested:

- rolling prelaunch reliability features
- recent-failure indicators
- recent short-window success rates
- regularized logistic models using `L1` and elastic-net penalties
- calibration and richer diagnostics

#### Outcome

This was a **clear positive result**.

The strongest model in that notebook was the regularized rolling-history logistic model, especially:

- `Logistic rolling L1 CV`

Its approximate chronological test performance was:

- balanced accuracy: `0.856`
- failure recall: `0.769`
- PR AUC: `0.635`

This is materially better than the earlier simpler chronological logistic baseline.

#### Interpretation

The main takeaway is that a **targeted expansion of prelaunch reliability history** worked much better than the broader, more diffuse feature-engineering pass from the earlier notebook.

That strongly supports the idea that the most valuable next improvements are still centered on:

- recent reliability history
- maturity / experience
- regularized linear models

Rather than moving to a more complex nonlinear model family.

### 2. `modeling_weather_V2.ipynb`: Base Model Vs Weather-Enhanced Subset

This notebook tested a different strategy:

- fit a broad base model on the full sample
- define a stricter “good weather coverage” subset
- compare a base model and a weather-enhanced model only within that cleaner subset

#### Outcome

This notebook produced **interesting but fragile** results.

The weather-subset models looked very strong on the subset test split, but the subset itself was much smaller and easier than the full sample. In the current run:

- the subset had only `243` rows
- the validation subset was especially weak
- some facilities contributed very few retained launches

In fact, the **weather-subset base logistic** often looked as strong as or stronger than the weather-enhanced variant on thresholded metrics.

#### Interpretation

This does **not** yet prove that a weather-subset strategy is superior.

What it does suggest is:

- weather may be more useful when restricted to high-quality matched rows
- subset-based weather modeling might be viable
- the current subset definition and validation design are too fragile to support a strong conclusion

So this direction is still worth exploring, but it should be treated as exploratory rather than established.

## What Looks Worth Trying Now

After these V2 results, the strongest next experiments are:

### 1. Push further on rolling-history features

This now looks like the best direction by a wide margin.

Worth trying:

- recent failure count over last `1`, `3`, `5`, and `10` launches
- time since last failure
- success rate over the last `3`, `5`, `10`, and `20` launches
- cadence features such as launches in the last `30`, `90`, and `365` days
- family/org/config/site volatility measures

### 2. Use blockwise chronological feature selection

Instead of adding many features at once, add blocks in order:

1. core reliability
2. recent reliability
3. vehicle / mission
4. weather
5. missingness / weather-quality variables

Only keep a block if it improves chronological validation performance.

This is likely better than large one-shot feature expansions.

### 3. Keep logistic regression central, but regularize aggressively

The V2 results make a stronger case for:

- `L1` logistic
- elastic-net logistic
- sparse, interpretable models

than for switching to more flexible nonlinear models.

At this point, logistic regression is not just the most interpretable model. It is also the strongest practical modeling family in the project.

### 4. Improve calibration and ranking diagnostics

Since the model increasingly looks useful as a **risk-ranking tool**, not just a classifier, it is worth expanding:

- calibration diagnostics
- top-decile / top-quintile capture rates
- lift charts
- ranking-oriented summaries of how failures concentrate in the highest predicted-risk buckets

### 5. Refine the weather-subset idea before trusting it

If the weather-subset path is revisited, the next step should not just be “add more weather features.”

Instead:

- redefine the subset more carefully
- ensure enough failures remain in validation and test
- compare sites separately if needed
- treat it as a controlled subset experiment, not a replacement for the full-sample model

### 6. Consider site-specific rolling-history models for the biggest sites

A strong next targeted experiment would be:

- one model for Cape / Kennedy launches
- one model for Vandenberg launches

using the rolling-history feature design from `modeling_V2.ipynb`

This would test whether the main successful V2 idea becomes even stronger when the model is not forced to pool different operational environments together.

## Updated Bottom Line

After the V2 notebook results, the current best-supported direction is:

1. keep chronological evaluation
2. build better recent-history reliability features
3. use regularized logistic regression as the main modeling family
4. treat weather as secondary or subset-specific until its coverage issue is handled more carefully

So the project has moved from “we should probably try richer reliability features” to “we now have direct evidence that richer rolling-history reliability features are one of the best improvements in the codebase so far.”
