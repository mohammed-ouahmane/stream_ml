from models.decision_tree import dt_param_selector
from models.kneighbors_classifier import knn_param_selector
from models.linear_regression import lir_param_selector
from models.logistic_regression import lor_param_selector
from models.neural_network import nn_param_selector
from models.random_forest_classifier import rf_param_selector
from models.regression_ridge import rd_param_selector
from models.svc import svc_param_selector
from models.regression_lasso import rl_param_selector
from models.regression_elastic_net import en_param_selector
from models.svr import svr_param_selector

def getAlgorims(df):
    classification = {'DecisionTreeClassifier': dt_param_selector,'kneighbors classifier': knn_param_selector,
                      'random forest classifie': rf_param_selector,'SVC': svc_param_selector, 'neural network': nn_param_selector, 'logistic regression': lor_param_selector,}
    regression = {'linear regression': lir_param_selector,'regression ridge': rd_param_selector, 'regression lasso' : rl_param_selector, 'regression Elastic Net' : en_param_selector,
                  'SVR': svr_param_selector}
    column_type = df['target'].dtype
    if column_type == 'object':
        return classification
    else:
        return regression
