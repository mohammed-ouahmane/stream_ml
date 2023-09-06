import numpy as np
import streamlit as st
from sklearn.linear_model import Ridge

def rd_param_selector():
    params = {}
    if st.session_state['on'] :
        alpha = st.sidebar.number_input("alpha", 0.0, 100.0, 10.0, 1.0)
        params =  {"alpha": alpha }
    model = Ridge(**params)
    return model
