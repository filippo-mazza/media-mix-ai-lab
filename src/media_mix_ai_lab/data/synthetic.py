"""Reproducible teaching data; coefficients are not estimates or causal evidence."""

import numpy as np
import pandas as pd

OUTLIER_POSITION = 65
DEFAULT_SEED = 42
MAX_WEEKLY_MEDIA_INVESTMENT = 10_000
OUTLIER_LIFT = 8_000


def generate_sample_data(n_weeks: int = 104, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return fresh weekly synthetic data with one explicitly labelled revenue spike.

    Spend and revenue use arbitrary currency units. The additive generating process
    intentionally omits real-world complexity; never use it for business decisions.
    """
    if isinstance(n_weeks, bool) or not isinstance(n_weeks, int) or n_weeks < 66:
        raise ValueError("n_weeks must be an integer of at least 66.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    week = np.arange(n_weeks)
    seasonality = np.sin(2 * np.pi * week / 52)
    total_media = np.floor(rng.uniform(3_000, MAX_WEEKLY_MEDIA_INVESTMENT, n_weeks) * 100) / 100
    channel_mix = rng.dirichlet((4.0, 2.5, 1.5), n_weeks)
    tv, search = (channel_mix[:, :2] * total_media[:, np.newaxis]).round(2).T
    social = (total_media - tv - search).round(2)
    price = rng.uniform(90, 110, n_weeks)
    promotions = rng.binomial(1, 0.2, n_weeks)
    revenue = (
        12_000
        + 0.55 * tv
        + 0.85 * search
        + 0.45 * social
        - 90 * (price - 100)
        + 1_500 * promotions
        + 2_000 * seasonality
        + rng.normal(0, 500, n_weeks)
    )
    revenue[OUTLIER_POSITION] += OUTLIER_LIFT
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON"),
            "revenue": revenue.round(2),
            "tv_spend": tv.round(2),
            "search_spend": search.round(2),
            "social_spend": social.round(2),
            "price_index": price.round(3),
            "promotions": promotions,
            "seasonality": seasonality.round(6),
            "is_injected_outlier": week == OUTLIER_POSITION,
        }
    )


def main() -> None:
    """Regenerate the committed sample from the repository root."""
    from pathlib import Path

    destination = Path("data/samples/weekly_marketing.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_data().to_csv(destination, index=False)
    print(f"Wrote synthetic data to {destination}")


if __name__ == "__main__":
    main()
