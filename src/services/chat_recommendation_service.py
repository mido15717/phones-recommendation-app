from __future__ import annotations

from difflib import get_close_matches
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from core.logger import logger
from helpers.LLM_Finder import LLMFinder
from helpers.config import get_settings
from models.chat import ChatMessage, ConversationTurn, LLMConfig, PhonePreferences
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
    You are a friendly, sharp phone-buying assistant for the Egyptian market. Your
    job is to figure out six preferences well enough to search a real phone
    catalog: budget (EGP), use_case, minimum storage (GB), minimum RAM (GB), brand
    preference, and network (4G/5G).

    MOST USERS DO NOT KNOW TECHNICAL SPECS. Never expect the user to say "128GB
    storage" or "8GB RAM" literally. Instead, infer specs from what they actually
    care about, the way a knowledgeable friend or phone-shop employee would:

    - "I want to take pictures and vlog" -> photography use case -> needs strong
    camera + decent storage for video (lean toward higher storage, e.g. 128–256GB)
    and enough RAM for smooth multitasking while recording (e.g. 8GB+).
    - "I play heavy games" -> gaming use case -> prioritize RAM (8GB+) and storage
    for large game installs (128GB+).
    - "just calls, WhatsApp, browsing" -> light use -> lower storage/RAM is fine
    (64GB, 4-6GB).
    - "I don't care about brand" / "whatever's good" -> mark brand complete with
    null value (genuinely no preference), do NOT ask again.
    - "I don't know" about storage/RAM specifically -> do NOT leave it null. Infer
    a sensible default from their stated use_case and budget instead, using your
    best judgment, and treat that slot as complete. You may briefly mention your
    assumption in the reply (e.g. "Since you're vlogging, I'll look for phones
    with plenty of storage for your videos.").
    - Only leave a value genuinely null when the user explicitly says they have no
    preference at all AND no other answer (use case, budget) gives you a
    reasonable way to infer one.

    Ask exactly ONE natural question at a time for whichever preference is still
    missing or unclear. Prioritize learning use_case and budget early, since they
    let you infer storage/RAM defaults even if the user never states specs
    directly. Never invent a specific phone or model at this stage — only infer
    preference values. When every slot is complete, briefly say you are finding
    suitable phones.

    IMPORTANT: `completed_slots` must contain ONLY these exact strings, one per
    preference you now consider settled (either from an explicit answer or a
    confident inference) — no other spelling or variant is recognized:
    "budget", "use_case", "min_storage", "min_ram", "brand", "network"

    A slot counts as complete once you have EITHER a concrete value OR a
    deliberate "no preference" — not simply because the user hasn't mentioned it
    yet.

    Valid data values (only pick values from these lists when inferring):
    - prices: {self.options['price_min']:.0f}–{self.options['price_max']:.0f} EGP
    - brands: {", ".join(self.options['brands'])}
    - storage GB: {", ".join(str(int(x)) for x in self.options['storage'])}
    - RAM GB: {", ".join(str(int(x)) for x in self.options['ram'])}
    - network: 4G, 5G, or no preference

    Current preferences: {preferences.model_dump()}
    Current use case: {use_case or 'unknown'}
    Completed slots: {completed_slots}

    Return a structured response. Preserve already-completed values unless the
    user explicitly changes them. For numeric RAM/storage values, always choose
    the nearest listed value to whatever you infer or the user states.
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
        llm_completed = set(turn.completed_slots) & self.REQUIRED_SLOTS
        completed = sorted(llm_completed)
        return turn.model_copy(update={"preferences": prefs, "completed_slots": completed})

    def _resolve_llm(self, llm_config: LLMConfig | None) -> BaseChatModel:
        if llm_config and (llm_config.api_key or llm_config.provider == "ollama"):
            if llm_config.provider == "openai":
                if not llm_config.api_key:
                    raise RuntimeError("OpenAI requires an API key.")
                llm = LLMFinder.connect_via_api(
                    api_key=llm_config.api_key,
                    model_name=llm_config.model_name or "gpt-4o-mini",
                    temperature=llm_config.temperature,
                    base_url=llm_config.base_url,
                )
            elif llm_config.provider == "ollama":
                if not llm_config.model_name:
                    raise RuntimeError("Ollama requires a model name.")
                from langchain_ollama import ChatOllama
                llm = ChatOllama(
                    model=llm_config.model_name,
                    temperature=llm_config.temperature,
                    base_url=llm_config.base_url,
                )
            else:
                llm = None

            if llm is None:
                raise RuntimeError("Could not connect with the supplied LLM credentials.")
            return llm

        if self.llm is None:
            raise RuntimeError(
                "No LLM configured. Enter an API key in the sidebar, or set "
                "AI_PROVIDER/API_KEY/LLM_MODEL in .env."
            )
        return self.llm

    def _invoke_turn(
        self,
        message: str,
        history: list[ChatMessage],
        preferences: PhonePreferences,
        use_case: str | None,
        completed_slots: list[str],
        llm_config: LLMConfig | None = None,
    ) -> ConversationTurn:
        llm = self._resolve_llm(llm_config)
        messages = [SystemMessage(content=self._system_prompt(preferences, use_case, completed_slots))]
        messages.extend(
            HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
            for item in history
        )
        messages.append(HumanMessage(content=message))
        try:
            result = llm.with_structured_output(ConversationTurn).invoke(messages)
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
            raise RuntimeError("Could not understand the phone request." f"{error}") from error

    def chat_turn(
        self,
        message: str,
        history: list[ChatMessage],
        preferences: PhonePreferences,
        use_case: str | None,
        completed_slots: list[str],
        top_k: int,
        llm_config: LLMConfig | None = None,
    ) -> dict:
        turn = self._invoke_turn(message, history, preferences, use_case, completed_slots, llm_config)
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