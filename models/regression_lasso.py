import numpy as np
from sklearn.linear_model import Lasso
import streamlit as st

def rl_param_selector():
    params = {}
    if st.session_state['on'] :
        alpha = st.sidebar.number_input("alpha", 0.00, 100.00, 0.10, 0.05)
        params =  {"alpha": alpha }
    model = Lasso (**params)
    return model