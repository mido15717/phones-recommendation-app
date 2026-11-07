from __future__ import annotations

from difflib import get_close_matches
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from core.logger import logger
from helpers.LLM_Finder import LLMFinder
from helpers.config import get_settings
from models.chat import ChatMessage, ConversationTurn, LLMConfig, PhonePreferences,FollowUpTurn, PhoneComparison
from services.lookup_service import ListingService
from services.recommendation_service import RecommendationService

EMPTY_COMPARISON: dict = {"summary": "", "top_pick": None, "top_pick_reason": None}

class ChatRecommendationService:
    """A multi-turn phone assistant that calls RAG only after preference collection."""

    REQUIRED_SLOTS = {"budget", "use_case", "min_storage", "min_ram", "brand", "network"}

    def __init__(
        self,
        recommendation_service: RecommendationService,
        grouped_df: pd.DataFrame,
        listing_service: "ListingService | None" = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.recommendation_service = recommendation_service
        self.llm = llm or LLMFinder.connect_from_settings(get_settings())
        self.options = self._build_options(grouped_df)
        self.listing_service = listing_service

    def _enrich_with_listings(self, phones: list[dict]) -> list[dict]:
        if not self.listing_service:
            return phones
        enriched = []
        for phone in phones:
            listings = self.listing_service.get_listings(
                brand=phone.get("brand", ""), model=phone.get("model", "")
            )
            enriched.append({**phone, "listings": listings})
        return enriched

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
    @staticmethod
    def _format_listings(phone: dict, limit: int = 3, include_url: bool = False) -> str:
        listings = phone.get("listings", [])
        if not listings:
            return "No live listings found."

        parts = []
        for listing in listings[:limit]:
            storage = listing.get("storage_gb")
            ram = listing.get("ram_gb")
            color = listing.get("color")
            price = listing.get("price_egp")
            seller = listing.get("seller_name") or "unknown seller"
            in_stock = listing.get("in_stock")

            storage_str = f"{storage:.0f}GB" if storage is not None else "storage unspecified"
            ram_str = f"{ram:.0f}GB RAM" if ram is not None else "RAM unspecified"
            color_str = f" {color}" if color else ""
            price_str = f"EGP {price:,.0f}" if price is not None else "price unavailable"
            stock_str = "in stock" if in_stock else "out of stock"

            line = f"{storage_str}/{ram_str}{color_str} — {price_str} at {seller} ({stock_str})"
            if include_url and listing.get("product_url"):
                line += f" — {listing['product_url']}"
            parts.append(line)

        return "; ".join(parts)

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

    def _normalise_preferences(self, prefs: PhonePreferences) -> PhonePreferences:
        prefs = prefs.model_copy()
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
        return prefs

    def _normalise(self, turn: ConversationTurn) -> ConversationTurn:
        prefs = self._normalise_preferences(turn.preferences)

        llm_completed = set(turn.completed_slots) & self.REQUIRED_SLOTS

        return turn.model_copy(
            update={
                "preferences": prefs,
                "completed_slots": sorted(llm_completed),
            }
        )
    
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
                    base_url=llm_config.base_url or "http://host.docker.internal:11434",
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
        llm = self._resolve_llm(llm_config)
        turn = self._invoke_turn(message, history, preferences, use_case, completed_slots, llm_config)
        complete = self.REQUIRED_SLOTS.issubset(turn.completed_slots)
        recommendations: list[dict] = []
        comparison = dict(EMPTY_COMPARISON)

        if complete:
            semantic_query = turn.use_case or message
            recommendations = self.recommendation_service.hybrid_retrieve(
                query=semantic_query,
                top_k=top_k,
                **turn.preferences.model_dump(),
            )
            recommendations = self._enrich_with_listings(recommendations)
            if recommendations:
                comparison = self._compare_phones(
                    phones=recommendations, preferences=turn.preferences,
                    use_case=turn.use_case, message=message, llm=llm,
                )
            else:
                turn = turn.model_copy(update={
                    "reply": (
                        "I couldn't find any phones matching all of your preferences "
                        f"(budget {turn.preferences.budget:.0f} EGP, brand {turn.preferences.brand or 'any'}, "
                        f"network {turn.preferences.network or 'any'}). Want to try a higher budget, "
                        "a different brand, or drop the network requirement?"
                    )
                })

        return {
            "reply": turn.reply,
            "preferences": turn.preferences.model_dump(),
            "use_case": turn.use_case,
            "completed_slots": turn.completed_slots,
            "complete": complete,
            "recommendations": recommendations,
            "comparison": comparison,
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
        phones = self._enrich_with_listings(phones)
        return {"filters": preferences.model_dump(), "recommendations": phones}

    def _followup_system_prompt(self, preferences: PhonePreferences, recommendations: list[dict]) -> str:
        catalog = "\n\n".join(
            f"{phone.get('brand')} {phone.get('model')}\n"
            f"Specs: {phone.get('spec_text') or 'No spec text available.'}\n"
            f"Listings: {self._format_listings(phone, include_url=True)}"
            for phone in recommendations
        )
        return f"""
    You are helping a user who has already received a shortlist of recommended
    phones. Every new message is one of exactly two intents:

    1. answer_question — the user asks about specs, pricing, stock, purchase
    links, OR asks you to compare, rank, or evaluate the phones ALREADY SHOWN
    below (e.g. "compare these", "which is better", "compare battery",
    "great performance over price"). ANY question about the phones already
    listed — including comparisons and evaluations — is answer_question.
    NEVER treat a comparison/evaluation request as refine_search, even if it
    mentions words like "performance" or "price". Answer using ONLY the specs
    and listings below. Never invent a price, URL, or spec not explicitly
    present. If pricing/link info isn't available, say so honestly. Keep
    `preferences` unchanged.

    2. refine_search — the user explicitly wants to see a DIFFERENT set of
    phones: a different budget, a different brand, more storage, etc. This is
    about changing search CRITERIA, not evaluating phones already shown.
    Adjust ONLY the preference(s) they're actually talking about:
    - "cheaper" / "a bit less" (no number given) -> lower budget by ~15-20%
    - "more expensive" / "show me pricier options" (no number) -> raise
        budget by ~15-20%
    - an explicit number/brand/spec -> use it exactly
    Leave every other preference exactly as it currently is. Return the FULL
    updated preferences object.

    Current preferences: {preferences.model_dump()}

    Phones currently shown to the user:
    {catalog}

    Decide the intent for this message and respond accordingly.
    """.strip()

    def handle_followup(
        self,
        message: str,
        history: list[ChatMessage],
        preferences: PhonePreferences,
        recommendations: list[dict],
        comparison: dict | None,
        top_k: int,
        llm_config: LLMConfig | None = None,
    ) -> dict:
        llm = self._resolve_llm(llm_config)
        current_comparison = comparison or dict(EMPTY_COMPARISON)
        messages = [SystemMessage(content=self._followup_system_prompt(preferences, recommendations))]
        messages.extend(
            HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
            for item in history
        )
        messages.append(HumanMessage(content=message))

        try:
            result = llm.with_structured_output(FollowUpTurn).invoke(messages)
            if isinstance(result, dict):
                result = FollowUpTurn(**result)
            if not isinstance(result, FollowUpTurn):
                raise TypeError("The LLM returned an unexpected follow-up format.")
        except Exception as error:
            logger.exception("Failed to process a follow-up turn.")
            raise RuntimeError(f"Could not understand the follow-up request. {error}") from error

        if result.intent == "refine_search":
            values = result.preferences.model_dump()
            previous_values = preferences.model_dump()
            for field, previous_value in previous_values.items():
                if values[field] is None:
                    values[field] = previous_value
            new_prefs = self._normalise_preferences(PhonePreferences(**values))
            semantic_query = message
            previous_ids = [p["model_id"] for p in recommendations]
            new_recommendations = self.recommendation_service.hybrid_retrieve(
                query=semantic_query, top_k=top_k, exclude_ids=previous_ids, **new_prefs.model_dump()
            )
            new_recommendations = self._enrich_with_listings(new_recommendations)
            new_comparison = (
                self._compare_phones(
                    phones=new_recommendations, preferences=new_prefs,
                    use_case=None, message=message, llm=llm,
                )
                if new_recommendations
                else dict(EMPTY_COMPARISON)
            )
            return {
                "reply": result.reply,
                "preferences": new_prefs.model_dump(),
                "recommendations": new_recommendations,
                "comparison": new_comparison,
                "intent": "refine_search",
            }

        return {
            "reply": result.reply,
            "preferences": preferences.model_dump(),
            "recommendations": recommendations,
            "comparison": current_comparison,
            "intent": "answer_question",
        }

    def _compare_phones(
    self,
    phones: list[dict],
    preferences: PhonePreferences,
    use_case: str | None,
    message: str,
    llm: BaseChatModel,
) -> dict:
        if not phones:
            return dict(EMPTY_COMPARISON)

        catalog_text = "\n\n".join(
            f"Option {index + 1}: {phone.get('brand')} {phone.get('model')}\n"
            f"Specs: {phone.get('spec_text') or 'No spec text available.'}\n"
            f"Listings: {self._format_listings(phone, include_url=True)}"
            for index, phone in enumerate(phones)
        )

        prompt = f"""
    You are comparing a shortlist of real phones for a specific user, based only
    on the specs and listings given below. Do not invent or assume any spec,
    price, or feature not present in the text.

    User's use case: {use_case or 'unspecified'}
    User's stated preferences: {preferences.model_dump()}
    User's original message: "{message}"

    Phone options:
    {catalog_text}

    Write a short, natural comparison (3-6 sentences) that highlights the key
    differences between these specific phones that matter most for this user's
    use case (e.g. camera quality, battery life, performance, display, storage,
    price — whichever are relevant and actually present in the specs/listings).
    Do not restate every spec verbatim; synthesize what's meaningfully different
    between them.

    Then pick ONE phone as the best overall fit for this user and briefly explain
    why, referencing the specific specs or listing details that make it the right
    choice for their stated needs.
    """.strip()

        try:
            result = llm.with_structured_output(PhoneComparison).invoke([SystemMessage(content=prompt)])
            if isinstance(result, dict):
                result = PhoneComparison(**result)
            if not isinstance(result, PhoneComparison):
                raise TypeError("The LLM returned an unexpected comparison format.")
            return result.model_dump()
        except Exception:
            logger.exception("Failed to compare retrieved phones.")
            return dict(EMPTY_COMPARISON)
