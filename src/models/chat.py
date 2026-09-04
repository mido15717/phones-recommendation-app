from typing import Literal

from pydantic import BaseModel, Field


class PhonePreferences(BaseModel):
    """Constraints extracted from a user's natural-language request."""

    budget: float | None = Field(default=None, ge=0)
    brand: str | None = None
    min_storage: float | None = Field(default=None, ge=0)
    min_ram: float | None = Field(default=None, ge=0)
    network: str | None = None
    category: str | None = None
    in_stock: bool | None = True


class ChatRecommendationRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    # UI filters take precedence over values inferred from natural language.
    preferences: PhonePreferences | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class ConversationTurn(BaseModel):
    reply: str
    preferences: PhonePreferences = Field(default_factory=PhonePreferences)
    use_case: str | None = None
    completed_slots: list[str] = Field(default_factory=list)


class ChatTurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    preferences: PhonePreferences = Field(default_factory=PhonePreferences)
    use_case: str | None = None
    completed_slots: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
