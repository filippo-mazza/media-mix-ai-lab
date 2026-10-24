# Media Mix AI Lab engineering conventions

This is an experimental Python 3.12 / Streamlit scientific workbench. Keep changes
small and tied to a story. Do not implement speculative modeling or AI features.

## Architecture

- `app.py` calls `src/media_mix_ai_lab/ui/app.py`; Streamlit and Plotly belong in UI.
- `data/` in the package owns synthetic generation and validation; repository
  `data/samples/` holds public synthetic fixtures only. Never commit private uploads.
- `core/` owns Pydantic model configurations; `modeling/` defines numerical engine
  contracts. `diagnostics/` and `optimization/` reserve future domain boundaries.
- `ai/` defines provider-neutral conversation and explicitly allowlisted, typed
  deterministic tools. No provider, API key, framework, or vector store is required.
- Results and provenance belong in numerical result objects. AI prose is separate.

## Commands (repository root)

```sh
uv sync --locked
uv run streamlit run app.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m media_mix_ai_lab.data.synthetic
```

Use `pyproject.toml` and commit `uv.lock`; do not add a redundant requirements file.
After dependency changes run `uv lock` and validate `uv sync --locked`. Add type
annotations for application functions, useful validation errors, and tests of
behavior. Test UI changes with Streamlit AppTest. Keep business logic out of UI.

## Statistical correctness

LLMs reason while deterministic tools compute. Never accept invented coefficients,
metrics, ROI, p-values, or predictions. No arbitrary generated Python execution.
Validate typed tool inputs, keep the allowlist explicit, and interpret only actual
results; typed narrative alone does not guarantee factual correctness.

Never silently remove outliers, sort, impute, or transform user input. Reject bad
inputs with actionable errors; make any future cleaning an explicit operation.
Preserve input snapshots/hashes, configurations, random seeds, training windows,
and runtime versions. Prevent temporal leakage and document log-domain, adstock,
control-variable, and uncertainty assumptions when those features are implemented.
Correlation is not causal attribution. Revenue/spend is not incremental ROI.
Synthetic examples must remain labelled. Keep disclaimers visible and never
present this personal project as an employer-endorsed decision system.
