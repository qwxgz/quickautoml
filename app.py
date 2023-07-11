import streamlit as st
import pandas as pd
import os

# Import Profiling capabilities
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Import ML things
from pycaret.classification import setup, compare_models, pull, save_model
# from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://tse2-mm.cn.bing.net/th/id/OIP-C.ql7qtUlFZWrq2MbK72jPFwHaFQ?pid=ImgDet&rs=1")
    st.title("自动化机器学习 AutoStreamML")
    choice = st.radio("功能浏览 Navigation", ["Upload", "Profiling", "AutoML", "Download"])
    st.info("该项目可运行于网页端，通过建立自动机器学习AutoML管道，能够快速对数据特征进行自动化分析和建模分析，非常方便和快捷。- drafted by qwx")
# st.write("Hello world!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("上传数据 Upload Data for AutoML")
    file = st.file_uploader("Upload Your Dataset from Button below")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        pass

if choice == "Profiling":
    st.title("自动化EDA数据探索分析 Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    pass

if choice == "AutoML":
    st.title("启动自动机器学习 Launch Auto Machine Learning")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("训练AutoML模型 Train AutoML Models"):
        st.info("模型训练中，请等待......Traing in Progress ")
        setup(df, target=target)
        setup_df = pull()  # check this
        st.info("自动机器学习的设置参数 AutoML Settings to Be Used:")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("AutoML所用的模型 These are the Models Used in AutoML")
        st.dataframe(compare_df)
        st.info("最佳模型 The Best Model is: ")
        st.info(best_model)
        save_model(best_model, 'best_model')
    pass

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("下载模型 Download the Model", f, "trained_ml_model.pkl")
    pass