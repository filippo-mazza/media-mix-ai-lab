"""Streamlit page for chronology-preserving model validation."""

import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.diagnostics.temporal import (
    ErrorMetrics,
    TemporalValidationResult,
    evaluate_temporal_holdout,
    same_holdout_period,
)
from media_mix_ai_lab.modeling.interfaces import ModelResult


def diagnostics_page() -> None:
    """Evaluate saved model specifications on a user-selected chronological holdout."""
    st.title("Diagnostics")
    st.write("Measure how saved model runs generalize to later weeks.")
    frame = st.session_state["dataset"]
    mapping = st.session_state["dataset_mapping"]
    model_runs = st.session_state.get("model_results", [])
    if not model_runs:
        st.info("Fit a model on the Models page before evaluating a holdout.")
        return
    if len(frame) < 2:
        st.warning("At least two weekly observations are needed for a temporal split.")
        return

    models = [ModelResult.model_validate_json(model_json) for model_json in model_runs]
    run_labels = [
        f"Fit {index} · {result.config.kind} · {result.dataset_sha256[:8]}"
        for index, result in enumerate(models, start=1)
    ]
    selected_label = st.selectbox("Stored model run", run_labels)
    selected_index = run_labels.index(selected_label)
    split_method = st.selectbox("Holdout selection", ["Holdout fraction", "Date cutoff"])
    split_options: dict[str, object] = {}
    if split_method == "Holdout fraction":
        split_options["holdout_fraction"] = st.slider(
            "Holdout fraction",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
        )
        split_label = f"{split_options['holdout_fraction']:.0%} holdout"
    else:
        cutoffs = frame[mapping["date"]].iloc[1:].tolist()
        cutoff = st.selectbox(
            "First holdout week",
            cutoffs,
            format_func=lambda value: value.strftime("%Y-%m-%d"),
        )
        split_options["cutoff"] = cutoff
        split_label = f"cutoff {cutoff.strftime('%Y-%m-%d')}"

    if st.button("Evaluate holdout"):
        try:
            evaluation = evaluate_temporal_holdout(
                frame,
                models[selected_index].config,
                **split_options,
            )
        except (TypeError, ValueError) as error:
            st.error(f"Holdout evaluation failed: {error}")
        else:
            st.session_state.setdefault("temporal_results", []).append(
                {
                    "run_label": selected_label,
                    "split_label": split_label,
                    "result_json": evaluation.model_dump_json(),
                }
            )

    evaluations = st.session_state.get("temporal_results", [])
    if not evaluations:
        return
    evaluation_labels = [
        f"Evaluation {index} · {item['run_label']} · {item['split_label']}"
        for index, item in enumerate(evaluations, start=1)
    ]
    st.subheader("Saved holdout evaluations")
    latest = len(evaluations) - 1
    display_label = st.selectbox(
        "Evaluation to inspect",
        evaluation_labels,
        index=latest,
    )
    display_index = evaluation_labels.index(display_label)
    _show_temporal_result(
        TemporalValidationResult.model_validate_json(evaluations[display_index]["result_json"]),
        evaluations[display_index]["run_label"],
    )

    if len(evaluations) >= 2:
        st.subheader("Compare two stored evaluations")
        left, right = st.columns(2)
        first_label = left.selectbox(
            "First evaluation", evaluation_labels, index=len(evaluations) - 2
        )
        second_label = right.selectbox("Second evaluation", evaluation_labels, index=latest)
        first = evaluations[evaluation_labels.index(first_label)]
        second = evaluations[evaluation_labels.index(second_label)]
        first_result = TemporalValidationResult.model_validate_json(first["result_json"])
        second_result = TemporalValidationResult.model_validate_json(second["result_json"])
        if not same_holdout_period(first_result, second_result):
            st.warning(
                "These evaluations use different holdout periods; compare the metrics with caution."
            )
        st.dataframe(
            pd.DataFrame(
                [
                    _comparison_row(first, first_result),
                    _comparison_row(second, second_result),
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "WAPE = sum of absolute errors ÷ sum of absolute observed outcomes; it is undefined "
        "when that denominator is zero. Log-log predictions are exponentiated to outcome "
        "units and represent a conditional geometric-scale prediction, not an arithmetic-mean "
        "correction. No causal attribution is estimated."
    )


def _show_temporal_result(result: TemporalValidationResult, run_label: str) -> None:
    st.markdown(f"**{run_label}**")
    st.caption(
        f"Training: {result.training_period[0][:10]} to {result.training_period[1][:10]} "
        f"({result.training_size} weeks) · Holdout: {result.holdout_period[0][:10]} to "
        f"{result.holdout_period[1][:10]} ({result.holdout_size} weeks)"
    )
    st.write("Errors in outcome units; training and holdout are reported separately.")
    metrics = pd.DataFrame(
        [
            _metric_row("Training", result.training_metrics),
            _metric_row("Holdout", result.holdout_metrics),
        ]
    )
    st.dataframe(metrics, hide_index=True, width="stretch")

    predictions = pd.DataFrame(
        [
            {
                "Week": row.date,
                "Observed": row.observed,
                "Predicted": row.predicted,
                "Residual": row.residual,
            }
            for row in result.predictions
        ]
    )
    observed_prediction = px.line(
        predictions.melt(
            id_vars="Week",
            value_vars=["Observed", "Predicted"],
            var_name="Series",
            value_name="Outcome",
        ),
        x="Week",
        y="Outcome",
        color="Series",
        title="Holdout: observed vs predicted",
    )
    st.plotly_chart(observed_prediction, width="stretch")
    st.plotly_chart(
        px.line(predictions, x="Week", y="Residual", title="Holdout residuals over time"),
        width="stretch",
    )

    autocorrelation, collinearity = st.columns(2)
    with autocorrelation:
        st.subheader("Residual autocorrelation")
        st.metric(
            "Durbin–Watson",
            "—" if result.durbin_watson is None else f"{result.durbin_watson:.4g}",
        )
        st.metric(
            "Ljung–Box p-value",
            "—" if result.ljung_box_p_value is None else f"{result.ljung_box_p_value:.4g}",
        )
        st.caption(
            "Computed on holdout residuals. Small samples and constant residuals make these "
            "diagnostics unavailable; neither test establishes model validity."
        )
    with collinearity:
        st.subheader("Training-design collinearity")
        st.metric("Condition number", f"{result.condition_number:.4g}")
        if result.variance_inflation_factors:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Predictor": predictor, "VIF": value}
                        for predictor, value in result.variance_inflation_factors.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("VIF was unavailable for this training sample.")
        st.caption(
            "Condition number depends on predictor units and scaling. VIF is computed for "
            "training predictors only and is not meaningful for a very short sample."
        )
    for warning in result.warnings:
        st.warning(warning)


def _metric_row(label: str, metrics: ErrorMetrics) -> dict[str, str]:
    return {
        "Period": label,
        "MAE": f"{metrics.mae:.4g}",
        "RMSE": f"{metrics.rmse:.4g}",
        "WAPE": "Undefined" if metrics.wape is None else f"{metrics.wape:.2%}",
    }


def _comparison_row(evaluation: dict[str, str], result: TemporalValidationResult) -> dict[str, str]:
    return {
        "Evaluation": evaluation["run_label"],
        "Holdout period": (f"{result.holdout_period[0][:10]} to {result.holdout_period[1][:10]}"),
        "MAE": f"{result.holdout_metrics.mae:.4g}",
        "RMSE": f"{result.holdout_metrics.rmse:.4g}",
        "WAPE": (
            "Undefined"
            if result.holdout_metrics.wape is None
            else f"{result.holdout_metrics.wape:.2%}"
        ),
    }
