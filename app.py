from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from recommender import load_default_recommender


st.set_page_config(page_title="Curated | Product recommendations", page_icon="✦", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #192522; --muted: #66736f; --mint: #d9f4e5; --coral: #ff8364; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: linear-gradient(145deg, #f8fbf7 0%, #eef6f0 55%, #fff7ef 100%); }
    .hero { padding: 2.3rem 0 1rem; }
    .eyebrow { color: #e26045; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .75rem; }
    .hero h1 { font-size: clamp(2.8rem, 6vw, 5.6rem); line-height: .96; margin: .4rem 0 1rem; max-width: 780px; }
    .hero p { color: var(--muted); font-size: 1.1rem; max-width: 600px; }
    .product-card { background: rgba(255,255,255,.72); border: 1px solid rgba(25,37,34,.1); padding: 1rem; height: 100%; box-shadow: 0 12px 30px rgba(35,55,44,.06); }
    .product-card h3 { margin: .65rem 0 .25rem; font-size: 1.08rem; }
    .product-card p { color: var(--muted); font-size: .88rem; line-height: 1.45; }
    .price { color: #e26045; font-weight: 700; font-size: 1.1rem; }
    .reason { color: #397653; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
    </style>
    """,
    unsafe_allow_html=True,
)

recommender = load_default_recommender()

st.markdown('<div class="hero"><div class="eyebrow">A quieter way to discover</div><h1>Find the next thing you will love.</h1><p>Choose something you like and let product details, taste patterns, and real ratings shape the shortlist.</p></div>', unsafe_allow_html=True)

with st.container(border=True):
    left, right = st.columns([2.2, 1])
    with left:
        selected_name = st.selectbox("Start with a product", recommender.product_names())
    with right:
        selected_user = st.selectbox("Personalize for", ["Everyone"] + sorted(recommender.ratings["user_id"].unique().tolist()))
    limit = st.slider("Recommendations", min_value=3, max_value=8, value=5)

user_id = None if selected_user == "Everyone" else selected_user
recommendations = recommender.recommend(selected_name, user_id=user_id, limit=limit)

st.write("")
header_left, header_right = st.columns([2, 1])
with header_left:
    st.subheader(f"Because you chose {selected_name}")
    st.caption("A blend of content similarity and collaborative signals")
with header_right:
    export_frame = pd.DataFrame([r.__dict__ for r in recommendations])
    export_buffer = BytesIO()
    export_frame.to_csv(export_buffer, index=False)
    st.download_button("Export shortlist as CSV", export_buffer.getvalue(), "recommendations.csv", "text/csv", use_container_width=True)

columns = st.columns(min(4, len(recommendations)))
for column, item in zip(columns * ((len(recommendations) + len(columns) - 1) // len(columns)), recommendations):
    with column:
        st.image(item.image_url, use_container_width=True)
        st.markdown(f'<div class="product-card"><div class="reason">{item.reason}</div><h3>{item.name}</h3><div class="price">₹{item.price:,.2f}</div><p>{item.description}</p><p><b>{item.category}</b> · match score {item.score:.0%}</p></div>', unsafe_allow_html=True)

st.divider()
st.caption("Demo data is included in data/. Replace products.csv and ratings.csv to use your own catalog.")
