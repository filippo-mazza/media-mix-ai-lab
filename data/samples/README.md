# Synthetic weekly marketing data

`weekly_marketing.csv` contains 104 observations generated with NumPy seed 42.
Regenerate from the repository root with:

```sh
uv run python -m media_mix_ai_lab.data.synthetic
```

Revenue is a noisy additive combination of random spend, price, promotions,
and annual seasonality. Row 65 (zero-based) has an additional 60,000 revenue
units and is labelled `is_injected_outlier`. No cleaning removes it.
See `src/media_mix_ai_lab/data/synthetic.py` for the complete generating process.
Currency units are arbitrary. This is teaching data, not evidence of real media effects.
