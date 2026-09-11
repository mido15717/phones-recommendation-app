import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

GREETING = (
    "Hi! I’ll help you find a phone. "
    "What budget are you working with in EGP?"
)


st.set_page_config(
    page_title="Phone Recommendation Chat",
    layout="wide",
)


st.markdown(
    """
    <style>
    .phone-card {
        padding: 18px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin: 12px 0;
        background: #fff;
    }

    .phone-name {
        font-size: 1.2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_conversation() -> None:
    """Reset all conversation state."""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": GREETING,
        }
    ]

    st.session_state.preferences = {
        "in_stock": True,
    }

    st.session_state.use_case = None
    st.session_state.completed_slots = []
    st.session_state.recommendations = None
    st.session_state.comparison = None

    # collecting -> post_recommendation
    st.session_state.phase = "collecting"


def render_phone(phone: dict) -> None:
    """Render a single recommended phone."""

    brand = phone.get("brand", "Unknown brand")
    model = phone.get("model", "Unknown model")

    # ---------------------------------------------------------
    # Price
    # ---------------------------------------------------------

    price_min = phone.get("price_min")
    price_max = phone.get("price_max")

    price = "Price unavailable"

    if price_min is not None:
        try:
            if price_max in (None, price_min):
                price = f"EGP {float(price_min):,.0f}"
            else:
                price = (
                    f"EGP {float(price_min):,.0f}"
                    f"–{float(price_max):,.0f}"
                )
        except (TypeError, ValueError):
            price = "Price unavailable"

    # ---------------------------------------------------------
    # Phone header
    # ---------------------------------------------------------

    st.markdown(
        f"""
        <div class="phone-card">
            <div class="phone-name">
                {brand} {model}
            </div>
            <div>
                {price}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Main specifications
    # ---------------------------------------------------------

    columns = st.columns(4)

    ram_options = phone.get("ram_options", [])
    storage_options = phone.get("storage_options", [])
    network_options = phone.get("network", [])

    columns[0].metric(
        "RAM",
        ", ".join(
            f"{item} GB"
            for item in ram_options
        )
        or "N/A",
    )

    columns[1].metric(
        "Storage",
        ", ".join(
            f"{item} GB"
            for item in storage_options
        )
        or "N/A",
    )

    columns[2].metric(
        "Network",
        ", ".join(
            str(item)
            for item in network_options
        )
        or "N/A",
    )

    score = phone.get("semantic_score")

    columns[3].metric(
        "Match score",
        f"{score:.3f}"
        if score is not None
        else "N/A",
    )

    # ---------------------------------------------------------
    # Specifications
    # ---------------------------------------------------------

    with st.expander("View specifications"):
        st.write(
            phone.get(
                "spec_text",
                "No specifications available.",
            )
        )

    # ---------------------------------------------------------
    # Concrete listings
    # ---------------------------------------------------------

    listings = phone.get("listings", [])

    if not listings:
        return

    st.write("**Where to buy:**")

    for listing in listings:
        storage = listing.get("storage_gb")
        ram = listing.get("ram_gb")
        color = listing.get("color") or ""

        # Build variant label
        if storage is not None:
            try:
                variant = f"{float(storage):.0f}GB"
            except (TypeError, ValueError):
                variant = f"{storage}GB"
        else:
            variant = "?"

        if ram is not None:
            try:
                variant += f" / {float(ram):.0f}GB RAM"
            except (TypeError, ValueError):
                variant += f" / {ram}GB RAM"

        if color:
            variant += f", {color}"

        # Listing information
        price_egp = listing.get("price_egp")
        seller_name = listing.get(
            "seller_name",
            "Unknown seller",
        )
        product_url = listing.get("product_url")
        in_stock = listing.get("in_stock", False)

        stock_label = (
            "In stock"
            if in_stock
            else "Out of stock"
        )

        # Price formatting
        if price_egp is not None:
            try:
                formatted_price = f"EGP {float(price_egp):,.0f}"
            except (TypeError, ValueError):
                formatted_price = "Price unavailable"
        else:
            formatted_price = "Price unavailable"

        # -----------------------------------------------------
        # Render listing
        # -----------------------------------------------------

        listing_text = (
            f"- **{variant}** — "
            f"**{formatted_price}** "
            f"at {seller_name} — {stock_label}"
        )

        if product_url:
            listing_text += f" — [View product]({product_url})"

        st.markdown(listing_text)


def render_comparison(comparison: dict | None) -> None:
    """Render comparison summary and top pick."""

    if not comparison:
        return

    summary = comparison.get("summary")

    if summary:
        st.subheader("How these compare")
        st.write(summary)

    top_pick = comparison.get("top_pick")

    if top_pick:
        top_pick_reason = comparison.get(
            "top_pick_reason",
            "",
        )

        if top_pick_reason:
            st.success(
                f"**Top pick:** {top_pick} — "
                f"{top_pick_reason}"
            )
        else:
            st.success(
                f"**Top pick:** {top_pick}"
            )


# =============================================================
# Initialize session state
# =============================================================

if "messages" not in st.session_state:
    reset_conversation()


# =============================================================
# Sidebar
# =============================================================

with st.sidebar:
    st.header("LLM settings")

    provider = st.selectbox(
        "Provider",
        ["openai", "ollama"],
    )

    api_key = None

    # ---------------------------------------------------------
    # OpenAI
    # ---------------------------------------------------------

    if provider == "openai":
        api_key = st.text_input(
            "API key",
            type="password",
        )

        model_name = st.text_input(
            "Model",
            value="gpt-4o-mini",
        )

    # ---------------------------------------------------------
    # Ollama
    # ---------------------------------------------------------

    else:
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/rag/local-models",
                timeout=10,
            )

            response.raise_for_status()

            local_models = response.json().get(
                "models",
                [],
            )

        except requests.RequestException:
            local_models = []

        if local_models:
            model_name = st.selectbox(
                "Model",
                local_models,
            )

        else:
            st.caption(
                "No local Ollama models found — "
                "is Ollama running?"
            )

            model_name = st.text_input(
                "Model (type manually)",
                value="llama3",
            )

    st.session_state.llm_config = {
        "provider": provider,
        "api_key": api_key or None,
        "model_name": model_name,
    }

    # ---------------------------------------------------------
    # Conversation settings
    # ---------------------------------------------------------

    st.header("Conversation settings")

    top_k = st.slider(
        "Results to show",
        min_value=1,
        max_value=20,
        value=5,
    )

    if st.button(
        "Start a new conversation",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()

    st.divider()

    completed = st.session_state.completed_slots

    st.caption(
        f"Preferences collected: "
        f"{len(completed)} / 6"
    )

    if completed:
        st.write(
            ", ".join(
                slot.replace("_", " ")
                for slot in completed
            )
        )


# =============================================================
# Main page
# =============================================================

st.title("Find your next phone")

st.caption(
    "Answer one short question at a time. "
    "Recommendations use only phones in the local dataset."
)


# =============================================================
# Conversation history
# =============================================================

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])


# =============================================================
# User input
# =============================================================

if user_message := st.chat_input("Type your answer…"):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.write(user_message)

    try:
        with st.chat_message("assistant"), st.spinner("Thinking…"):

            # =================================================
            # Post-recommendation follow-up
            # =================================================

            if st.session_state.phase == "post_recommendation":

                payload = {
                    "message": user_message,
                    "history": st.session_state.messages[:-1],
                    "preferences": st.session_state.preferences,
                    "recommendations": (
                        st.session_state.recommendations
                        or []
                    ),
                    "comparison": (
                        st.session_state.comparison
                        or {}
                    ),
                    "top_k": top_k,
                    "llm_config": st.session_state.llm_config,
                }

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/rag/chat/followup",
                    json=payload,
                    timeout=60,
                )

                response.raise_for_status()

                data = response.json()

                st.write(data["reply"])

                # Update state
                st.session_state.preferences = (
                    data["preferences"]
                )

                st.session_state.recommendations = (
                    data["recommendations"]
                )

                # -------------------------------------------------
                # BUG FIX #1
                #
                # answer_question responses may not contain
                # "comparison".
                #
                # Keep the existing comparison in that case.
                # -------------------------------------------------

                st.session_state.comparison = data.get(
                    "comparison",
                    st.session_state.comparison,
                )

            # =================================================
            # Preference collection
            # =================================================

            else:

                payload = {
                    "message": user_message,
                    "history": st.session_state.messages[:-1],
                    "preferences": st.session_state.preferences,
                    "use_case": st.session_state.use_case,
                    "completed_slots": (
                        st.session_state.completed_slots
                    ),
                    "comparison": st.session_state.get("comparison"),
                    "top_k": top_k,
                    "llm_config": st.session_state.llm_config,
                }

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/rag/chat/turn",
                    json=payload,
                    timeout=60,
                )

                response.raise_for_status()

                data = response.json()

                st.write(data["reply"])

                # Update collected preferences
                st.session_state.preferences = (
                    data["preferences"]
                )

                st.session_state.use_case = (
                    data["use_case"]
                )

                st.session_state.completed_slots = (
                    data["completed_slots"]
                )

                # -------------------------------------------------
                # BUG FIX #2
                #
                # Capture the comparison returned when the
                # initial recommendation process completes.
                # -------------------------------------------------

                if data["complete"]:

                    st.session_state.recommendations = (
                        data["recommendations"]
                    )

                    st.session_state.comparison = data.get(
                        "comparison"
                    )

                    st.session_state.phase = (
                        "post_recommendation"
                    )

        # =====================================================
        # Save assistant response
        # =====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data["reply"],
            }
        )

        st.rerun()

    except requests.RequestException as error:

        detail = (
            "Check that the FastAPI server and "
            "LLM configuration are running."
        )

        if error.response is not None:
            try:
                detail = error.response.json().get(
                    "detail",
                    detail,
                )
            except ValueError:
                pass

        st.error(
            f"I couldn't continue the conversation: "
            f"{detail}"
        )


# =============================================================
# Recommendations
# =============================================================

if st.session_state.recommendations is not None:

    st.divider()

    recommendations = (
        st.session_state.recommendations
    )

    # ---------------------------------------------------------
    # No results
    # ---------------------------------------------------------

    if not recommendations:

        st.info(
            "I collected your preferences, but no "
            "in-stock phones matched them. Try a new "
            "conversation with fewer constraints."
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    else:

        # -----------------------------------------------------
        # BUG FIX #3
        #
        # Render comparison before the individual phones.
        # -----------------------------------------------------

        comparison = (
            st.session_state.get("comparison")
            or {}
        )

        render_comparison(comparison)

        # -----------------------------------------------------
        # Phone cards
        # -----------------------------------------------------

        st.subheader(
            f"Best matches ({len(recommendations)})"
        )

        for phone in recommendations:
            render_phone(phone)