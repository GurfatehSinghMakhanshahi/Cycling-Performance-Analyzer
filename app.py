import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Cycling Performance Analyzer', layout = 'wide')

st.sidebar.title("Navigate Here!")

page = st.sidebar.selectbox(
    "Choose the Page",
    ["Home", "Performance Analyzer"]
)


@st.cache_data
def load_d():
    return pd.read_csv("data//cycling_clean.csv")

@st.cache_resource
def load_model():
    return joblib.load("D:\\Cycling Performance Analysis\\Cycling_Report\\Cycling_Report\\analysis\\mix_model.pkl")

data = load_d()
model = load_model()


if page == "Home":
    st.title("Cycling Performance Analyzer")
    st.markdown("""
    ### Welcome!
                
    This app is used to predict **rider performance points** based on characteristics of rider and stage.
    It is built using a **Mixed Linear Model** on cycling performance data.
                
    #### Features:
                
    1. **Performance Analyzer** -- Analyse rider points based on class and stage type
    2. **Data Overview** -- Visualization of statistics
    3. **Smart UI** -- Easy navigation with streamlit design
                
    **Instructions :**
                
    - Use the sidebar to navigate between pages
    - Enter your input parameters in the *Performance Analyzer* page
    - View dataset summaries in the *Data Overview* page
                """)
    
else:
    st.title("Performance AnalYzer")

    st.sidebar.subheader("Input Parameters")

    rider_c = st.sidebar.selectbox("Select rider class", data["rider_class"].unique())
    stage_c = st.sidebar.selectbox("Select stage class", data['stage_class'].unique())

    st.markdown(f"**Selected Rider Class :** {rider_c}")
    st.markdown(f"**Selected Stage Class :** {stage_c}")

    input_df = pd.DataFrame({
        "rider_class": [rider_c],
        "stage_class": [stage_c],
        "rider_id": [0]
    })

    prediction = model.predict(input_df)[0]
    st.subheader(f"Prediction Points: **{prediction: .2f}**")


st.markdown("""
<hr style='border:1px solid gray'>
<p style='text-align:center; color:gray;'>                
""", unsafe_allow_html = True)