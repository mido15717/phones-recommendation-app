from __future__ import annotations

from difflib import get_close_matches
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from core.logger import logger
from helpers.LLM_Finder import LLMFinder
from helpers.config import get_settings
from models.chat import ChatMessage, ConversationTurn, PhonePreferences
from services.recommendation_service import RecommendationService


class ChatRecommendationService:
    """A multi-turn phone assistant that calls RAG only after preference collection."""

    REQUIRED_SLOTS = {"budget", "use_case", "min_storage", "min_ram", "brand", "network"}

    def __init__(
        self,
        recommendation_service: RecommendationService,
        grouped_df: pd.DataFrame,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.recommendation_service = recommendation_service
        self.llm = llm or LLMFinder.connect_from_settings(get_settings())
        self.options = self._build_options(grouped_df)

    @staticmethod
    def _build_options(df: pd.DataFrame) -> dict[str, Any]:
        parse = RecommendationService._parse_list
        storage = sorted({float(x) for value in df["storage_options"] for x in parse(value)})
        ram = sorted({float(x) for value in df["ram_options"] for x in parse(value)})
        return {
            "brands": sorted(df["brand"].dropna().astype(str).unique().tolist()),
            "storage": storage,
            "ram": ram,
            "networks": ["4G", "5G"],
            "price_min": float(pd.to_numeric(df["price_min"], errors="coerce").min()),
            "price_max": float(pd.to_numeric(df["price_max"], errors="coerce").max()),
        }

    def _system_prompt(
        self,
        preferences: PhonePreferences,
        use_case: str | None,
        completed_slots: list[str],
    ) -> str:
        return f"""
You are a concise phone-buying assistant for the Egyptian market. Collect these
six preferences through natural conversation: budget (EGP), use_case,
minimum storage (GB), minimum RAM (GB), brand preference, and network (4G/5G).

Ask exactly ONE question for one missing preference at a time. A user may say
"no preference" for brand or network; mark that slot complete and leave its
value null. Never invent a phone or recommend a specific model at this stage.
When every slot is complete, briefly say you are finding suitable phones.

Valid data values:
- prices: {self.options['price_min']:.0f}–{self.options['price_max']:.0f} EGP
- brands: {", ".join(self.options['brands'])}
- storage GB: {", ".join(str(int(x)) for x in self.options['storage'])}
- RAM GB: {", ".join(str(int(x)) for x in self.options['ram'])}
- network: 4G, 5G, or no preference

Current preferences: {preferences.model_dump()}
Current use case: {use_case or 'unknown'}
Completed slots: {completed_slots}

Return a structured response. Preserve already-completed values unless the
user explicitly changes them. Mark a slot complete only when the user gives a
value or explicitly has no preference. For numeric RAM/storage values that do
not exactly exist, choose the nearest listed value.
""".strip()

    def _normalise(self, turn: ConversationTurn) -> ConversationTurn:
        prefs = turn.preferences.model_copy()
        if prefs.brand:
            match = get_close_matches(prefs.brand, self.options["brands"], n=1, cutoff=0.6)
            prefs.brand = match[0] if match else None
        for field, options_key in (("min_storage", "storage"), ("min_ram", "ram")):
            value = getattr(prefs, field)
            if value is not None:
                setattr(prefs, field, min(self.options[options_key], key=lambda option: abs(option - value)))
        if prefs.network:
            network = prefs.network.strip().upper()
            prefs.network = network if network in self.options["networks"] else None
        completed = sorted(set(turn.completed_slots) & self.REQUIRED_SLOTS)
        return turn.model_copy(update={"preferences": prefs, "completed_slots": completed})

    def _invoke_turn(
        self,
        message: str,
        history: list[ChatMessage],
        preferences: PhonePreferences,
        use_case: str | None,
        completed_slots: list[str],
    ) -> ConversationTurn:
        if self.llm is None:
            raise RuntimeError(
                "LLM is not configured. Set AI_PROVIDER, API_KEY (for OpenAI), and LLM_MODEL in .env."
            )
        messages = [SystemMessage(content=self._system_prompt(preferences, use_case, completed_slots))]
        messages.extend(
            HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
            for item in history
        )
        messages.append(HumanMessage(content=message))
        try:
            result = self.llm.with_structured_output(ConversationTurn).invoke(messages)
            if isinstance(result, dict):
                result = ConversationTurn(**result)
            if not isinstance(result, ConversationTurn):
                raise TypeError("The LLM returned an unexpected conversation format.")
            result = self._normalise(result)
            # Structured output may omit earlier values. Keep a completed slot
            # unless the model supplies a replacement value for it.
            values = result.preferences.model_dump()
            previous_values = preferences.model_dump()
            for field, previous_value in previous_values.items():
                if field in completed_slots and values[field] is None:
                    values[field] = previous_value
            return result.model_copy(update={"preferences": PhonePreferences(**values)})
        except Exception as error:
            logger.exception("Failed to process a phone-chat turn.")
            raise RuntimeError("Could not understand the phone request.") from error

    def chat_turn(
        self,
        message: str,
        history: list[ChatMessage],
        preferences: PhonePreferences,
        use_case: str | None,
        completed_slots: list[str],
        top_k: int,
    ) -> dict:
        turn = self._invoke_turn(message, history, preferences, use_case, completed_slots)
        complete = self.REQUIRED_SLOTS.issubset(turn.completed_slots)
        recommendations: list[dict] = []
        if complete:
            semantic_query = " ".join(
                [item.content for item in history if item.role == "user"] + [message, turn.use_case or ""]
            )
            recommendations = self.recommendation_service.hybrid_retrieve(
                query=semantic_query,
                top_k=top_k,
                **turn.preferences.model_dump(),
            )
        return {
            "reply": turn.reply,
            "preferences": turn.preferences.model_dump(),
            "use_case": turn.use_case,
            "completed_slots": turn.completed_slots,
            "complete": complete,
            "recommendations": recommendations,
        }

    def recommend(
        self,
        message: str,
        top_k: int = 5,
        explicit_preferences: PhonePreferences | None = None,
    ) -> dict:
        """Retain the one-shot API for clients that do not need a conversation."""
        preferences = explicit_preferences or PhonePreferences()
        phones = self.recommendation_service.hybrid_retrieve(
            query=message, top_k=top_k, **preferences.model_dump()
        )
        return {"filters": preferences.model_dump(), "recommendations": phones}
