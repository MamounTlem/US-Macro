import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from fredapi import Fred
import datetime
from PIL import Image



# ---- CONFIG ----
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Sidebar background color and padding */
        section[data-testid="stSidebar"] {
            background-color: #e6f0fa;  /* Light corporate blue */
            padding: 1.5rem;
        }

        /* Sidebar text styling */
        .sidebar .stTextInput, .sidebar .stDateInput, .sidebar .stSelectbox {
            font-size: 14px;
        }

        /* Optional: shrink spacing between widgets */
        .sidebar .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* Optional: bold section headers */
        .sidebar .stExpanderHeader {
            font-weight: bold;
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# Add at the top of your app
logo = Image.open("logo.png")
st.columns([1, 3, 1])[1].image(logo, use_container_width=True)

st.markdown(
    "<h1 style='text-align: center; color: #003366;'> US Macro Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")
# ---- FRED SETUP ----
fred = Fred(api_key="b06a8f9099a23667b944f77d7264ff5b")


# ---- SIDEBAR ----
st.sidebar.title("Dashboard Options")
with st.sidebar.expander("📅 Date & Frequency", expanded=True):
    start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
    freq_choice = st.selectbox("Frequency", ["Quarterly", "Yearly"])






freq_map = {
    "Quarterly": "Q",
    "Yearly": "A"
}
freq_code = freq_map[freq_choice]

with st.sidebar.expander("➕ Add Custom Indicator"):
    new_label = st.text_input("Label (e.g. 'Unemployment Rate')", "")
    new_code = st.text_input("FRED Code (e.g. 'UNRATE')", "")
    impact = st.radio("Does a high value mean...", ["Good", "Bad"], index=0)
    if "custom_indicators" not in st.session_state:
        st.session_state.custom_indicators = {}
    if "custom_inverses" not in st.session_state:
        st.session_state.custom_inverses = []
    if st.sidebar.button("Add Indicator"):
        if new_label and new_code:
            try:
                test_series = fred.get_series(new_code, observation_start=start_date)
                if test_series.empty:
                    st.sidebar.warning(f"No data found for code '{new_code}'.")
                else:
                    st.session_state.custom_indicators[new_label] = new_code
                    if impact == "Bad" and new_label not in st.session_state.custom_inverses:
                        st.session_state.custom_inverses.append(new_label)
            except Exception as e:
                st.sidebar.warning(f"Failed to add: {e}")
        else:
            st.sidebar.warning("Please fill both fields.")

if st.sidebar.button("🔄 Reset Custom Indicators"):
    st.session_state.custom_indicators = {}
    st.session_state.custom_inverses = []

in_indicators = {
    "OECD Composite Leading Econ Index": "USALOLITOAASTSAM",
    "Manufacturers New Orders": "AMTMNO",
    "Initial Claims": "ICSA",
    "Consumer sentiment": "UMCSENT",
    "Building Permits": "PERMIT",
    "Freight Index": "FRGSHPUSM649NCIS",
    "SLOOS tightening": "DRTSCILM",
}

indicators = {**in_indicators, **st.session_state.custom_indicators}

# Merge with custom indicators


inverse_indicators = ["Initial Claims", "SLOOS tightening"] + st.session_state.custom_inverses


# ---- FETCH & PROCESS DATA ----
@st.cache_data(show_spinner=False)
def fetch_data(start_date, freq_code, indicators):
    frames = []
    successful_labels = []
    failed_labels = []

    for name, code in indicators.items():
        try:
            series = fred.get_series(code, observation_start=start_date)
            series = series.to_frame(name)
            series.index = pd.to_datetime(series.index)

            # Resample based on user choice
            if freq_code == 'Q':
                resampled = series.resample('Q').last()
            elif freq_code == 'A':
                resampled = series.resample('A').last()
            else:
                resampled = series

            frames.append(resampled)
            successful_labels.append(name)
        except Exception as e:
            failed_labels.append(name)
            st.warning(f"{name} failed to fetch: {e}")

    if failed_labels:
        st.warning(f"These indicators failed and won't show up: {', '.join(failed_labels)}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    return combined.dropna(), successful_labels

raw_df, valid_indicators = fetch_data(start_date, freq_code, indicators)
raw_df = raw_df.dropna(how='any')  # to avoid rows with missing data

# ---- NORMALIZATION ----
norm_df = raw_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
norm_df[inverse_indicators] *= -1
clipped_df = norm_df.clip(-2, 2)

# ---- FORMAT LABELS ----
# Set label frequency depending on data density
if freq_code == "Q":
    x_labels = clipped_df.index.to_period("Q").strftime("Q%q'%Y")
    xtick_freq = 1
elif freq_code == "A":
    x_labels = clipped_df.index.strftime("%Y")
    xtick_freq = 1



# ---- GLOBAL HEATMAP ----
st.subheader("General Heatmap")

fig, ax = plt.subplots(figsize=(18, 8))
norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
sns.heatmap(clipped_df.T, cmap="RdYlGn", annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.5,
            cbar_kws={'label': 'Z-score'}, norm=norm, ax=ax)
ax.set_xticks(np.arange(0, len(x_labels), xtick_freq) + 0.5)
ax.set_xticklabels(x_labels[::xtick_freq], rotation=45, fontsize=8)
ax.set_yticklabels(clipped_df.columns, fontsize=9)
st.pyplot(fig)

# ---- INDIVIDUAL VIEW ----
st.subheader("Indicator Detail View")
if raw_df.empty:
    st.error("No data available. Try changing the start date or frequency.")
    st.stop()

selected = st.selectbox("Select indicator", valid_indicators)


col1, col2 = st.columns(2)

with col1:
    st.markdown("**Raw Time Series**")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    raw_df[selected].plot(ax=ax2, title=selected, color='blue')
    st.pyplot(fig2)

with col2:
    st.markdown("**Normalized Strip Chart**")
    fig3, ax3 = plt.subplots(figsize=(10, 1.5))
    sns.heatmap(clipped_df[[selected]].T, cmap="RdYlGn", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5, cbar_kws={'label': 'Z-score'},
                norm=norm, ax=ax3)
    ax3.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax3.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax3.set_yticklabels([selected], fontsize=9)
    st.pyplot(fig3)

# ---- DOWNLOADS ----
st.subheader("Download Data")
st.download_button("Download Raw Data (CSV)", data=raw_df.to_csv().encode(), file_name="raw_data.csv", mime="text/csv")
st.download_button("Download Normalized Data (CSV)", data=clipped_df.to_csv().encode(), file_name="normalized_data.csv", mime="text/csv")
