import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


def knn_param_selector():
    params = {}
    if st.session_state['on'] :
        n_neighbors = st.sidebar.number_input("n_neighbors", 5, 20, 5, 1)
        metric = st.sidebar.selectbox(
        "metric", ("minkowski", "euclidean", "manhattan", "chebyshev", "mahalanobis")
        )

        params = {"n_neighbors": n_neighbors, "metric": metric}

    model = KNeighborsClassifier(**params)
    return model