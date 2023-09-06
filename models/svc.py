from sklearn import svm
import streamlit as st
from sklearn.svm import SVC


def svc_param_selector():
    params = {}
    if st.session_state['on'] :
        C = st.sidebar.number_input("C", 0.01, 100.0, 1.0, 1.0)
        kernel = st.sidebar.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid"))
        params = {"C": C, "kernel": kernel}
    model = SVC(**params)
    return model