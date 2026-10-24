"""Provider-neutral conversation and explicitly allowed deterministic tools.

No model SDK, remote calls, code execution, or automatic tool dispatch.
"""

from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from media_mix_ai_lab.data.validation import NUMERIC_COLUMNS, validate_dataset


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tool: Literal["summarize_column"] = "summarize_column"
    column: Literal[
        "revenue",
        "tv_spend",
        "search_spend",
        "social_spend",
        "price_index",
        "promotions",
        "seasonality",
    ]


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tool: Literal["summarize_column"] = "summarize_column"
    column: str
    count: int = Field(gt=0)
    mean: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat


def summarize_column(data: pd.DataFrame, request: SummaryRequest) -> SummaryResult:
    """The initial allowlisted tool: compute exact descriptive statistics locally."""
    validate_dataset(data)
    if request.column not in NUMERIC_COLUMNS:
        raise ValueError("Unsupported summary column.")
    values = data[request.column]
    return SummaryResult(
        column=request.column,
        count=len(values),
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Interpretation(BaseModel):
    """Narrative referencing actual tool results; never a numerical authority.

    A future integration must validate claims against supplied results before
    display. A typed schema alone cannot guarantee factual narrative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    limitations: tuple[str, ...]


class ConversationProvider(Protocol):
    def respond(
        self, messages: tuple[Message, ...], results: tuple[SummaryResult, ...]
    ) -> Interpretation:
        """Interpret supplied results; use explicit validated tools for computation."""
        ...
