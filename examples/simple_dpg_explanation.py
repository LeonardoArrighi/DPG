import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
#from dpg.core import DecisionPredicateGraph
#from dpg.explainer import DPGExplainer

from dpg.core import DecisionPredicateGraph
from dpg.explainer import DecisionPredicateGraphExplainer as DPGExplainer

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg.core import DecisionPredicateGraph


# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Fit the model
model = RandomForestClassifier(n_estimators=5, random_state=42)
model.fit(X, y)

dpg = DecisionPredicateGraph(
    model=model,
    feature_names=iris.feature_names,
    target_names= iris.target_names
,
    perc_var=0.001,
    decimal_threshold=2,
    n_jobs=1
)
dpg.fit(X)
dpg_explainer = DPGExplainer(dpg)
dpg_explainer.plot_class_bounds()
dpg_explainer.plot_communities()

dpg_explainer.plot_class_bounds_lines()



