# Decision Predicate Graph (DPG)

<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/DPG.png" width="300" />
</p>


DPG is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparency and explainability challenges.

DPG is a graph structure that captures the tree-based ensemble model and learned dataset details, preserving the relations among features, logical decisions, and predictions towards emphasising insightful points.
DPG enables graph-based evaluations and the identification of model decisions towards facilitating comparisons between features and their associated values while offering insights into the entire model.
DPG provides descriptive metrics that enhance the understanding of the decisions inherent in the model, offering valuable insights.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/custom_l2.jpg?raw=true" width="600" />
</p>

---

## The structure
The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/example.png?raw=true" width="600" />
</p>

## Metrics
The graph-based nature of DPG provides significant enhancements in the direction of a complete mapping of the ensemble structure.
| Property     | Definition | Utility |
|--------------|------------|---------|
| _Constraints_  | The intervals of values for each feature obtained from all predicates connected by a path that culminates in a given class. | Calculate the classification boundary values of each feature associated with each class. |
| _Betweenness centrality_ | Quantifies the fraction of all the shortest paths between every pair of nodes of the graph passing through the considered node. | Identify potential bottleneck nodes that correspond to crucial decisions. |
| _Local reaching centrality_ | Quantifies the proportion of other nodes reachable from the local node through its outgoing edges. | Assess the importance of nodes similarly to feature importance, but enrich the information by encompassing the values associated with features across all decisions. |
| _Community_ | A subset of nodes of the DPG which is characterised by dense interconnections between its elements and sparse connections with the other nodes of the DPG that do not belong to the community. | Understanding the characteristics of nodes to be assigned to a particular community class, identifying predominant predicates, and those that play a marginal role in the classification process. |


|Constraints | Betweenness centrality | Local reaching centrality | Community|
|------------|------------|--------------|--------------------|
![](https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/example_constraints.png) | ![](https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/example_bc.png) | ![](https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/example_lrc.png) | ![](https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/example_community.png) |
|Constraints(Class 1) = val3 < F1 ≤ val1, F2 ≤ val2 | BC(F2 ≤ val2) = 4/24 | LRC(F1 ≤ val1) = 6 / 7 | Community(Class 1) = F1 ≤ val1, F2 ≤ val2 |

---
## Installation

To install DPG locally, first clone the repository:

```bash
git clone https://github.com/LeonardoArrighi/DPG.git
cd DPG
```

Then, install the DPG library in development mode using `pip`:
```bash
pip install -e .
```

Alternatively, if using `pip directly`:
```bash
pip install git+https://github.com/LeonardoArrighi/DPG.git
```

---

## Example usage (Python)

You can also try DPG directly inside a Jupyter Notebook. Here's a minimal working example:

Here is a minimal example of how to use DPG with a trained model and a CSV dataset (e.g., `datasets/custom.csv`) using `scikit-learn` and the DPG API:

```python
# Example in Jupyter Notebook
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dpg.core import DecisionPredicateGraph
from dpg.visualizer import plot_dpg
from metrics.nodes import NodeMetrics
from metrics.graph import GraphMetrics

# Load dataset (last column assumed to be target)
df = pd.read_csv("datasets/custom.csv", index_col=0)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train simple Random Forest
model = RandomForestClassifier(n_estimators=10, random_state=27)
model.fit(X, y)

# Create DPG
feature_names = X.columns.tolist()
class_names = np.unique(y).astype(str).tolist()
dpg = DecisionPredicateGraph(
    model=model,
    feature_names=feature_names,
    target_names=class_names
)
dot = dpg.fit(X.values)
dpg_model, nodes_list = dpg.to_networkx(dot)

# Extract and visualize
dpg_metrics = GraphMetrics.extract_graph_metrics(dpg_model, nodes_list,target_names=np.unique(y_train).astype(str).tolist())
df = NodeMetrics.extract_node_metrics(dpg_model, nodes_list)
plot_dpg("dpg_output.png", dot, df_nodes, dpg_metrics, save_dir="datasets", communities=True, class_flag=True)
```
#### Output:
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/dpg_output.png_communities.png?raw=true" width="600" />
</p>

#### CLI scripts
The library contains two different scripts to apply DPG:
- `run_dpg_standard.py`: with this script it is possible to test DPG on a standard classification dataset provided by `sklearn` such as `iris`, `digits`, `wine`, `breast cancer`, and `diabetes`.
- `run_dpg_custom.py`: with this script it is possible to apply DPG to your classification dataset, specifying the target class.

#### DPG implementation
The library also contains two other essential scripts:
- `core.py` contains all the functions used to calculate and create the DPG and the metrics.
- `visualizer.py` contains the functions used to manage the visualization of DPG.

#### Output
The DPG output, through `run_dpg_standard.py` or `run_dpg_custom.py`, produces several files:
- the visualization of DPG in a dedicated environment, which can be zoomed and saved;
- a `.txt` file containing the DPG metrics;
- a `.csv` file containing the information about all the nodes of the DPG and their associated metrics;
- a `.txt` file containing the Random Forest statistics (accuracy, confusion matrix, classification report)

## Easy usage
Usage: `python run_dpg_standard.py --dataset <dataset_name> --n_learners <integer_number> --pv <threshold_value> --t <integer_number> --model_name <str_model_name> --dir <save_dir_path> --plot --save_plot_dir <save_plot_dir_path> --attribute <attribute> --communities --class_flag`
Where:
- `dataset` is the name of the standard classification `sklearn` dataset to be analyzed;
- `n_learners` is the number of base learners for the Random Forest;
- `pv` is the threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees;
- `t` is the decimal precision of each feature;
- `model_name` is the name of the `sklearn` model chosen to perform classification (`RandomForestClassifier`,`BaggingClassifier`,`ExtraTreesClassifier`,`AdaBoostClassifier` are currently available);
- `dir` is the path of the directory to save the files;
- `plot` is a store_true variable which can be added to plot the DPG;
- `save_plot_dir` is the path of the directory to save the plot image;
- `attribute` is the specific node metric which can be visualized on the DPG;
- `communities` is a store_true variable which can be added to visualize communities on the DPG;
- `class_flag` is a store_true variable which can be added to highlight class nodes.
  
Disclaimer: `attribute` and `communities` can not be added together, since DPG supports just one of the two visualizations.

The usage of `run_dpg_custom.py` is similar, but it requires another parameter:
- `target_column`, which is the name of the column to be used as the target variable;
- while `ds` is the path of the directory where the dataset is.

#### Example `run_dpg_standard.py`
Some examples can be appreciated in the `examples` folder: https://github.com/LeonardoArrighi/DPG/tree/main/examples

In particular, the following DPG is obtained by transforming a Random Forest with 5 base learners, trained on Iris dataset.
The used command is `python run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --dir examples --plot --save_plot_dir examples`.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/iris_bl5_perc0.001_dec2.png" width="800" />
</p>

The following visualizations are obtained using the same parameters as the previous example, but they show two different metrics: _Community_ and _Betweenness centrality_.
The used command for showing communities is `python run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --dir examples --plot --save_plot_dir examples --communities`.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/iris_bl5_perc0.001_dec2_communities.png" width="800" />
</p>

The used command for showing a specific property is `python run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --dir examples --plot --save_plot_dir examples --attribute "Betweenness centrality" --class_flag`.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/dpg_image_examples/iris_bl5_perc0.001_dec2_Betweennesscentrality.png" width="800" />
</p>

***
## Citation
If you use this for research, please cite. Here is an example BibTeX entry:

```bibtex
@inproceedings{arrighi2024dpg,
  title={Decision Predicate Graphs: Enhancing Interpretability in Tree Ensembles},
  author={Arrighi, Leonardo and Pennella, Luca and Marques Tavares, Gabriel and Barbon Junior, Sylvio},
  booktitle={World Conference on Explainable Artificial Intelligence},
  pages={311--332},
  year={2024},
  isbn = {978-3-031-63797-1},
  doi = {10.1007/978-3-031-63797-1_16},
  publisher = {Springer Nature Switzerland},
}
