import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression


def lor_param_selector():
    params = {}
    if st.session_state['on'] :
        solver = st.sidebar.selectbox(
        "solver", options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
        )

        if solver in ["newton-cg", "lbfgs", "sag"]:
            penalties = ["l2", "none"]

        elif solver == "saga":
            penalties = ["l1", "l2", "none", "elasticnet"]

        elif solver == "liblinear":
            penalties = ["l1"]

        penalty = st.sidebar.selectbox("penalty", options=penalties)
        C = st.sidebar.number_input("C", 0.1, 2.0, 1.0, 0.1)
        C = np.round(C, 3)
        max_iter = st.sidebar.number_input("max_iter", 100, 2000, step=50, value=100)

        params = {"solver": solver, "penalty": penalty, "C": C, "max_iter": max_iter}

    model = LogisticRegression(**params)
    return model