import streamlit as st
import pandas as pd
import pycaret as pc
import os

# Import Profiling capabilities
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Import ML things
from pycaret.classification import setup, compare_models, pull, save_model
# from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://tse2-mm.cn.bing.net/th/id/OIP-C.ql7qtUlFZWrq2MbK72jPFwHaFQ?pid=ImgDet&rs=1")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "AutoML", "Download"])
    st.info("This is a AutoML pipeline using Streamlit, Pandas Profiling, Pycaret and very easy to use!")
# st.write("Hello world!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for AutoML")
    file = st.file_uploader("Upload Your Dataset from Button below")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        pass

if choice == "Profiling":
    st.title("AutoEDA - Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    pass

if choice == "AutoML":
    st.title("Machine Learning go ")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train AutoML Models"):
        setup(df, target=target)
        setup_df = pull()  # check this
        st.info("The AutoML Settings to Be Used:")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("These are the Models Used in AutoML")
        st.dataframe(compare_df)
        st.info("The Best Model is: ")
        st.info(best_model)
        save_model(best_model, 'best_model')
    pass

if choice == "Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download the Model", f, "trained_ml_model.pkl")
    pass