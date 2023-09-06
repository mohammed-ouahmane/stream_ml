import numpy as np
import streamlit as st
from sklearn.svm import SVR

def svr_param_selector():
    params = {}
    if st.session_state['on']:
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, 1.0, 0.01)
        epsilon = st.sidebar.number_input("Epsilon", 0.01, 1.0, 0.1, 0.01)
        params = {"kernel": kernel, "C": C, "epsilon": epsilon}
    model = SVR(**params)
    return model
