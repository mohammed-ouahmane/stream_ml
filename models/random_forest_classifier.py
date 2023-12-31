import streamlit as st
from sklearn.ensemble import RandomForestClassifier


def rf_param_selector():
    params = {}
    if st.session_state['on'] :
        criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"])
        n_estimators = st.sidebar.number_input("n_estimators", 50, 300, 100, 10)
        max_depth = st.sidebar.number_input("max_depth", 1, 50, 5, 1)
        min_samples_split = st.sidebar.number_input("min_samples_split", 1, 20, 2, 1)
        max_features = st.sidebar.selectbox("max_features", [None, "auto", "sqrt", "log2"])

        params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
        }

    model = RandomForestClassifier(**params)
    return model