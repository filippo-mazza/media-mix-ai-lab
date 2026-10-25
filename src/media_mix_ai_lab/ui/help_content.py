"""Central glossary for concise, in-context scientific help."""

import streamlit as st

HELP_TEXT = {
    "weekly_data": (
        "Use one row per ordered week. Keep the source units and dates unchanged; the app "
        "rejects invalid weekly sequences instead of silently sorting or imputing them."
    ),
    "date_column": (
        "Choose the column containing unique weekly dates; rows must already be chronological."
    ),
    "target_column": (
        "The numeric outcome being modeled, in its original units (for example, weekly revenue)."
    ),
    "channel_columns": (
        "Choose media spend columns in consistent weekly units. Media coefficients are "
        "conditional associations, not causal ROI."
    ),
    "control_columns": (
        "Optional numeric covariates in their original units. Controls adjust associations "
        "but do not by themselves remove confounding."
    ),
    "model_log_log": (
        "Log-log OLS natural-logs the outcome and media signals, which must be positive; controls "
        "stay in original units. A media slope is a conditional elasticity, not a causal effect."
    ),
    "model_specification": (
        "Linear OLS (the default) uses outcome and predictors in original units. Log-log OLS "
        "natural-logs the positive outcome and media signals, while controls remain in original "
        "units; its media slopes are conditional elasticities. Neither model establishes causality."
    ),
    "outcome": (
        "Select the numeric outcome to explain; reported errors and linear coefficients use its "
        "units."
    ),
    "media_predictors": (
        "Selected media channels enter together, so each coefficient is conditional on the other "
        "selected media and controls. Mapped channels start selected."
    ),
    "optional_controls": (
        "Controls remain in their original units, including in log-log models. They can change "
        "media estimates; including them does not establish causal identification."
    ),
    "adstock_method": (
        "None (the default) keeps spend unchanged. Geometric adstock has a recursive tail; delayed "
        "geometric uses a finite lag window whose peak is controlled by theta."
    ),
    "adstock_alpha": (
        "Dimensionless decay: geometric uses 0 ≤ alpha < 1 and delayed geometric uses "
        "0 < alpha < 1. The Explore slider is 0–0.9; delayed-geometric zero is raised to 0.1. "
        "In Models, defaults are 0.5 and 0.7; Explore opens at 0.5. Higher alpha generally means "
        "slower decay; geometric half-life is log(0.5)/log(alpha) weeks."
    ),
    "adstock_initial_state": (
        "Starting carryover signal in the same units as media spend; default 0 assumes no "
        "pre-sample carryover."
    ),
    "adstock_max_lag": (
        "Delayed-geometric window length in weeks (0–12 here; default 8); spend older than this "
        "window does not contribute."
    ),
    "adstock_theta": (
        "Delayed-geometric peak lag in weeks, from 0 through the maximum lag (default 2). "
        "It shifts "
        "the largest lag weight; it is not a causal delay estimate."
    ),
    "adstock_history": (
        "Optional prior weekly spend values in the channel's units, oldest first. Leave blank "
        "to assume zero prehistory; values are not inferred."
    ),
    "adstock_normalization": (
        "Raw weights preserve their total scale; sum-to-one weights rescale the lag weights to "
        "sum to 1. Raw is the default; normalization changes signal units and can change "
        "coefficient scale."
    ),
    "intercept": (
        "Adds a constant baseline term (enabled by default). Without it, R² uses an uncentered "
        "convention and is not directly comparable to the usual centered R²."
    ),
    "sign_expectation": (
        "A review hypothesis, not a constraint or fitting target; the default is usually positive. "
        "Choose either sign when a negative association is plausible."
    ),
    "sample_counts": (
        "Counts refer to source rows. The current fitting policy retains all valid observations; "
        "nonpositive logged values are reported and cause log-log fitting to fail, not drop rows."
    ),
    "sample_size": (
        "Number of observations used in this fit; it is not the number of model parameters."
    ),
    "r_squared": (
        "In-sample fit summary: with an intercept, centered R² is the share of outcome variation "
        "explained by fitted values. Without an intercept this app uses uncentered R²; high R² "
        "does not establish causal lift."
    ),
    "adjusted_r_squared": (
        "R² adjusted for sample size and parameter count. It can help compare fitted "
        "specifications, "
        "but does not validate assumptions or causal claims."
    ),
    "residual_variance": ("Average squared in-sample residual scale, in squared outcome units."),
    "residuals": (
        "A residual is observed minus predicted outcome, in outcome units. Patterns over time can "
        "reveal systematic prediction errors; influence is not attribution."
    ),
    "coefficient_interpretation": (
        "A beta is a coefficient conditional on this specification. Linear media betas use outcome "
        "units per signal unit; log-log media betas are conditional elasticities."
    ),
    "statistical_evidence": (
        "Confidence intervals show a range compatible with the fitted model and assumptions. A "
        "p-value measures how surprising data this extreme are under a zero-coefficient null; it "
        "is not the probability that a coefficient is correct. Time dependence can invalidate "
        "classical inference."
    ),
    "multicollinearity": (
        "Highly overlapping predictors can make individual coefficients unstable. Condition number "
        "depends on units/scaling; VIF and thresholds are diagnostic heuristics, not automatic "
        "failures."
    ),
    "chronological_split": (
        "A holdout uses later contiguous weeks and never shuffles rows. The training prefix fits "
        "coefficients; changing the period changes what the reported errors mean."
    ),
    "holdout_fraction": (
        "Reserve this fraction of the latest observations for evaluation (0.1–0.5; default 0.2). "
        "Earlier rows train the model; no random split is used."
    ),
    "first_holdout_week": (
        "The selected date is the first holdout week; earlier rows train the model and this week "
        "and later rows are evaluated."
    ),
    "train_validation_test": (
        "Fit estimates parameters; candidates are ranked by validation RMSE, and the untouched "
        "final test estimates performance after selection. Reusing it makes it no longer untouched."
    ),
    "error_metrics": (
        "MAE is mean absolute error; RMSE is root mean squared error and penalizes large misses "
        "more; both use outcome units. WAPE divides total absolute error by total absolute "
        "observed "
        "outcome and is undefined when that denominator is zero."
    ),
    "adstock_half_life": (
        "For geometric 0 < alpha < 1, half-life is log(0.5)/log(alpha) weeks: the recursive "
        "signal's remaining weight halves over this interval. At alpha 0, decay is immediate."
    ),
    "optimization_objective": (
        "Candidates are ranked by chronological validation RMSE, a predictive-error objective—not "
        "by coefficient significance or causal impact. The final test is reserved for an explicit "
        "later evaluation."
    ),
    "candidate_budget": (
        "Maximum number of parameter combinations in the bounded search (1–100; default 20). A "
        "larger budget costs more fits but does not guarantee a better model."
    ),
    "candidate_status": (
        "A Not feasible or failed status means a valid score could not be produced under the data, "
        "split, or model constraints; it is not a poor-performance score. Small samples can make "
        "a split or fit infeasible."
    ),
    "influence_glossary": (
        "Influence diagnostics are not currently computed here. DFBETAS measures how a case shifts "
        "a coefficient; Cook's distance summarizes its joint influence; leverage measures unusual "
        "predictor position; studentized residuals scale its outcome error. Cutoffs are "
        "heuristics, "
        "and influence is not attribution."
    ),
    "outlier_policy": (
        "The app does not automatically remove, winsorize, or down-weight outliers; observations "
        "remain in the fit. Influence or sensitivity is not causal attribution, and any future "
        "exclusion should be explicit, justified, and compared with the full-data result."
    ),
}


def help_heading(title: str, topic: str, *, key: str) -> None:
    """Render a result heading with a keyboard- and touch-operable help popover."""
    heading, help_button = st.columns([12, 1], vertical_alignment="center")
    heading.subheader(title)
    with help_button.popover("?", key=f"help_{key}", help=f"Show help for {title}"):
        st.write(HELP_TEXT[topic])
