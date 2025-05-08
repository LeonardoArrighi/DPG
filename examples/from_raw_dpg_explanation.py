import ast
from dpg.explainer import DecisionPredicateGraphExplainer as DPGExplainer

# Read and parse the raw file (not as CSV)
with open("/home/barbon/Python/DPG/examples/temp/dpg_all.txt", "r") as f:
    raw_data = ast.literal_eval(f.read())

# Create explainer instance from raw dictionary
dpg_explainer = DPGExplainer.from_raw_data(raw_data)
dpg_explainer.plot_class_bounds()
dpg_explainer.plot_communities()
dpg_explainer.plot_class_bounds_lines()
