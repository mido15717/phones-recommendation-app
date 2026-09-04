import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
GREETING = "Hi! I’ll help you find a phone. What budget are you working with in EGP?"


st.set_page_config(page_title="Phone Recommendation Chat", page_icon="📱", layout="wide")

st.markdown(
    """
    <style>
    .phone-card { padding: 18px; border: 1px solid #e5e7eb; border-radius: 12px;
                  margin: 12px 0; background: #fff; }
    .phone-name { font-size: 1.2rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_conversation() -> None:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.preferences = {"in_stock": True}
    st.session_state.use_case = None
    st.session_state.completed_slots = []
    st.session_state.recommendations = None


def render_phone(phone: dict) -> None:
    brand = phone.get("brand", "Unknown brand")
    model = phone.get("model", "Unknown model")
    price_min = phone.get("price_min")
    price_max = phone.get("price_max")
    price = "Price unavailable"
    if price_min is not None:
        price = f"EGP {price_min:,.0f}" if price_max in (None, price_min) else f"EGP {price_min:,.0f}–{price_max:,.0f}"

    st.markdown(f'<div class="phone-card"><div class="phone-name">{brand} {model}</div>{price}</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("RAM", ", ".join(f"{item} GB" for item in phone.get("ram_options", [])) or "N/A")
    columns[1].metric("Storage", ", ".join(f"{item} GB" for item in phone.get("storage_options", [])) or "N/A")
    columns[2].metric("Network", ", ".join(str(item) for item in phone.get("network", [])) or "N/A")
    score = phone.get("semantic_score")
    columns[3].metric("Match score", f"{score:.3f}" if score is not None else "N/A")
    with st.expander("View specifications"):
        st.write(phone.get("spec_text", "No specifications available."))


if "messages" not in st.session_state:
    reset_conversation()

with st.sidebar:
    st.header("Conversation settings")
    top_k = st.slider("Results to show", min_value=1, max_value=20, value=5)
    if st.button("Start a new conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    st.divider()
    completed = st.session_state.completed_slots
    st.caption(f"Preferences collected: {len(completed)} / 6")
    if completed:
        st.write(", ".join(slot.replace("_", " ") for slot in completed))

st.title("📱 Find your next phone")
st.caption("Answer one short question at a time. Recommendations use only phones in the local dataset.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])

if user_message := st.chat_input("Type your answer…"):
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    payload = {
        "message": user_message,
        "history": st.session_state.messages[:-1],
        "preferences": st.session_state.preferences,
        "use_case": st.session_state.use_case,
        "completed_slots": st.session_state.completed_slots,
        "top_k": top_k,
    }
    try:
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            response = requests.post(f"{API_BASE_URL}/api/v1/rag/chat/turn", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            st.write(data["reply"])
        st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
        st.session_state.preferences = data["preferences"]
        st.session_state.use_case = data["use_case"]
        st.session_state.completed_slots = data["completed_slots"]
        if data["complete"]:
            st.session_state.recommendations = data["recommendations"]
        st.rerun()
    except requests.RequestException as error:
        detail = "Check that the FastAPI server and LLM configuration are running."
        if error.response is not None:
            try:
                detail = error.response.json().get("detail", detail)
            except ValueError:
                pass
        st.error(f"I couldn’t continue the conversation: {detail}")

if st.session_state.recommendations is not None:
    st.divider()
    recommendations = st.session_state.recommendations
    if not recommendations:
        st.info("I collected your preferences, but no in-stock phones matched them. Try a new conversation with fewer constraints.")
    else:
        st.subheader(f"Best matches ({len(recommendations)})")
        for phone in recommendations:
            render_phone(phone)
