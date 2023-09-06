import numpy as np
import streamlit as st
from sklearn.linear_model import ElasticNet

def en_param_selector():
    params = {}
    if st.session_state['on']:
        alpha = st.sidebar.number_input("alpha", 0.0, 1.0, 0.5, 0.01)
        l1_ratio = st.sidebar.number_input("L1 Ratio (mixing parameter)", 0.0, 1.0, 0.5, 0.01)
        params = {"alpha": alpha, "l1_ratio": l1_ratio}
    model = ElasticNet(**params)
    return model