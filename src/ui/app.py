import os
import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Phone Recommendation System",
    page_icon="📱",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .phone-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
        background-color: #ffffff;
    }

    .phone-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .phone-brand {
        font-size: 15px;
        color: #666;
        margin-bottom: 12px;
    }

    .price {
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
    }

    .score {
        font-size: 14px;
        color: #555;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.title("📱 Phone Recommendation System")

st.write(
    "Describe the phone you're looking for and we'll find "
    "the most relevant phones for you."
)


# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.header("🔎 Filters")

budget = st.sidebar.number_input(
    "Maximum budget",
    min_value=0,
    value=0,
    step=1000,
)

brand = st.sidebar.text_input(
    "Brand",
    placeholder="e.g. Samsung",
)

min_ram = st.sidebar.number_input(
    "Minimum RAM (GB)",
    min_value=0,
    value=0,
    step=1,
)

min_storage = st.sidebar.number_input(
    "Minimum storage (GB)",
    min_value=0,
    value=0,
    step=32,
)

network = st.sidebar.selectbox(
    "Network",
    [
        "Any",
        "4G",
        "5G",
    ],
)

category = st.sidebar.text_input(
    "Category",
    placeholder="e.g. Smartphone",
)

in_stock = st.sidebar.checkbox(
    "Only show phones in stock",
    value=True,
)


# ============================================================
# Search
# ============================================================

query = st.text_input(
    "What phone are you looking for?",
    placeholder=(
        "e.g. I want a powerful phone for gaming "
        "and photography"
    ),
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=20,
    value=5,
)


# ============================================================
# Search button
# ============================================================

if st.button(
    "🔍 Find Phones",
    type="primary",
    use_container_width=True,
):

    if not query.strip():

        st.warning(
            "Please describe the phone you're looking for."
        )

    else:

        # ----------------------------------------------------
        # Build API parameters
        # ----------------------------------------------------

        params = {
            "query": query,
            "top_k": top_k,
        }

        if budget > 0:
            params["budget"] = budget

        if brand.strip():
            params["brand"] = brand.strip()

        if min_ram > 0:
            params["min_ram"] = min_ram

        if min_storage > 0:
            params["min_storage"] = min_storage

        if network != "Any":
            params["network"] = network

        if category.strip():
            params["category"] = category.strip()

        params["in_stock"] = in_stock


        # ----------------------------------------------------
        # Call FastAPI
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Finding the best phones for you..."
            ):

                response = requests.get(
                    f"{API_BASE_URL}/api/v1/rag/search",
                    params=params,
                    timeout=30,
                )

                response.raise_for_status()

                data = response.json()


            # ------------------------------------------------
            # Extract results
            # ------------------------------------------------

            results = data.get(
                "results",
                data if isinstance(data, list) else []
            )


            if not results:

                st.info(
                    "No phones matched your requirements."
                )

            else:

                st.success(
                    f"Found {len(results)} recommendations."
                )


                # --------------------------------------------
                # Display cards
                # --------------------------------------------

                for phone in results:

                    brand_name = phone.get(
                        "brand",
                        "Unknown brand"
                    )

                    model = phone.get(
                        "model",
                        "Unknown model"
                    )

                    price_min = phone.get(
                        "price_min"
                    )

                    price_max = phone.get(
                        "price_max"
                    )

                    score = phone.get(
                        "semantic_score"
                    )

                    ram = phone.get(
                        "ram_options",
                        []
                    )

                    storage = phone.get(
                        "storage_options",
                        []
                    )

                    networks = phone.get(
                        "network",
                        []
                    )

                    stock = phone.get(
                        "in_stock",
                        False
                    )

                    # ----------------------------------------
                    # Card
                    # ----------------------------------------

                    st.markdown(
                        '<div class="phone-card">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        <div class="phone-title">
                            {brand_name} {model}
                        </div>

                        <div class="phone-brand">
                            {brand_name}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                    # ----------------------------------------
                    # Information columns
                    # ----------------------------------------

                    col1, col2, col3, col4 = st.columns(4)


                    with col1:

                        if price_min is not None:

                            if price_max is not None:

                                st.metric(
                                    "Price",
                                    f"{price_min:,.0f} - "
                                    f"{price_max:,.0f}"
                                )

                            else:

                                st.metric(
                                    "Price",
                                    f"{price_min:,.0f}"
                                )


                    with col2:

                        st.write("**RAM**")

                        if ram:
                            st.write(
                                ", ".join(
                                    f"{x} GB"
                                    for x in ram
                                )
                            )
                        else:
                            st.write("N/A")


                    with col3:

                        st.write("**Storage**")

                        if storage:
                            st.write(
                                ", ".join(
                                    f"{x} GB"
                                    for x in storage
                                )
                            )
                        else:
                            st.write("N/A")


                    with col4:

                        st.write("**Network**")

                        if networks:
                            st.write(
                                ", ".join(
                                    str(x)
                                    for x in networks
                                )
                            )
                        else:
                            st.write("N/A")


                    # ----------------------------------------
                    # Bottom information
                    # ----------------------------------------

                    score_col, stock_col = st.columns(2)


                    with score_col:

                        if score is not None:

                            st.write(
                                f"🎯 Semantic score: "
                                f"{score:.3f}"
                            )


                    with stock_col:

                        if stock:
                            st.success("✓ In stock")
                        else:
                            st.error("Out of stock")


                    # ----------------------------------------
                    # Specs
                    # ----------------------------------------

                    with st.expander(
                        "View specifications"
                    ):

                        st.write(
                            phone.get(
                                "spec_text",
                                "No specifications available."
                            )
                        )


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )


        except requests.RequestException as error:

            st.error(
                f"Could not reach the API: {error}"
            )