"""
Range Trading Bot — multi-page Streamlit app.
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Range Trading Bot", layout="wide")

pg = st.navigation([
    st.Page("pages/how_it_works.py", title="How It Works", icon=":material/school:", url_path="how-it-works", default=True),
    st.Page("pages/experiments.py", title="Experiments", icon=":material/compare_arrows:", url_path="experiments"),
    st.Page("pages/backtest.py", title="Backtest", icon=":material/candlestick_chart:", url_path="backtest"),
])

pg.run()
