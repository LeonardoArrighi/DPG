# Decision Predicate Graph (DPG)
DPG is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparency and explainability challenges.

DPG is a graph structure that captures the tree-based ensemble model and learned dataset details, preserving the relations among features, logical decisions, and predictions towards emphasising insightful points.
DPG enables graph-based evaluations and the identification of model decisions towards facilitating comparisons between features and their associated values while offering insights into the entire model.
DPG provides descriptive metrics that enhance the understanding of the decisions inherent in the model, offering valuable insights.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/custom_l2.jpg?raw=true" width="600" />
</p>

## The structure
The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/example.png?raw=true" width="400" />
</p>

## Metrics
The graph-based nature of DPG provides significant enhancements in the direction of a complete mapping of the ensemble structure.
| Property     | Definition | Utility |
|--------------|------------|---------|
| Constraints  | The intervals of values for each feature obtained from all predicates connected by a path that culminates in a given class. | Calculate the classification boundary values of each feature associated with each class. |
|Betweenness Centrality | Quantifies the fraction of all the shortest paths between every pair of nodes of the graph passing through the considered node. | Identify potential bottleneck nodes that correspond to crucial decisions. |
| Local Reaching Centrality | Quantifies the proportion of other nodes reachable from the local node through its outgoing edges. | Assess the importance of nodes similarly to feature importance, but enrich the information by encompassing the values associated with features across all decisions. |
| Community | A subset of nodes of the DPG which is characterised by dense interconnections between its elements and sparse connections with the other nodes of the DPG that do not belong to the community. | Understanding the characteristics of nodes to be assigned to a particular community class, identifying predominant predicates, and those that play a marginal role in the classification process. |


|Constraints | Betweenness Centrality | Local Reaching Centrality | Community|
|------------|------------|--------------|--------------------|
![](https://github.com/LeonardoArrighi/DPG/blob/main/examples/example_constraints.png) | ![](https://github.com/LeonardoArrighi/DPG/blob/main/examples/example_bc.png) | ![](https://github.com/LeonardoArrighi/DPG/blob/main/examples/example_lrc.png) | ![](https://github.com/LeonardoArrighi/DPG/blob/main/examples/example_community.png) |
|Constraints(Class 1) = val3 < F1 ≤ val1, F2 ≤ val2 | BC(F2 ≤ val2) = 4/24 | LRC(F1 ≤ val1) = 6 / 7 | Community(Class 1) = F1 ≤ val1, F2 ≤ val2 |

## The DPG library

#### Main script
The library contains two different scripts to apply DPG:
- `dpg_standard.py`: with this script it is possible to test DPG on a standard classification dataset provided by `sklearn` such as `iris`, `digits`, `wine`, `breast cancer`, and `diabetes`.
- `dpg_custom.py`: with this script it is possible to apply DPG to your classification dataset, specifying the target class.

#### Tree-based ensemble model: Random Forest
Random Forest, an example of a tree-based ensemble model, is already implemented within the scripts used by DPG. 

Specifically, the model is within `sklearn_standard_dpg.py`/`sklearn_custom_dpg.py`, the scripts used to manage the dataset, train the model, apply DPG, and apply the metrics.
Some Random Forest parameters cannot be modified outside the script where they are defined due to implementation choice.

#### Metrics and visualization
The library also contains two other essential scripts:
- `core.py` contains all the functions used to calculate and create the DPG and the metrics.
- `visualizer.py` contains the functions used to manage the visualization of DPG.

#### Output
The DPG application, through `dpg_standard.py` or `dpg_custom.py`, produces several files:
- the visualization of DPG in a dedicated environment, which can be zoomed and saved;
- a `.txt` file containing the DPG metrics;
- a `.csv` file containing the information about all the nodes of the DPG and their associated metrics;
- a `.txt` file containing the Random Forest statistics (accuracy, confusion matrix, classification report)

## Easy usage
Usage: `python dpg_standard.py --ds <dataset_name> --l <integer_number> --pv <threshold_value> --t <integer_number> --dir <save_dir_path> --plot --save_plot_dir <save_plot_dir_path> --attribute <attribute> --communities --class_flag`
Where:
- `ds` is the name of the standard classification `sklearn` dataset to be analyzed;
- `l` is the number of base learners for the Random Forest;
- `pv` is the threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees;
- `t` is the decimal precision of each feature;
- `dir` is the path of the directory to save the files;
- `plot` is a store_true variable which can be added to plot the DPG;
- `save_plot_dir` is path of the directory to save the plot image;
- `attribute` is the specific node metric which can be visualized on the DPG;
- `communities` is a store_true variable which can be added to visualize communities on the DPG;
- `class_flag` is a store_true variable which can be added to highlight class nodes.
  
Disclaimer: `attribute` and `communities` can not be added together, since DPG supports just one of the two visualizations.


The usage of `dpg_custom.py` is similar, but it requires another parameter:
- `target_column`, which is the name of the column to be used as the target variable;
- while `ds` is the path of the directory where the dataset is.

***
## Citation
If you use this for research, please cite. Here is an example BibTeX entry:


```
@misc{arrighi2024dpg,
      title={Decision Predicate Graphs: Enhancing Interpretability in Tree Ensembles}, 
      author={Leonardo Arrighi and Luca Pennella and Gabriel Marques Tavares and Sylvio Barbon Junior},
      year={2024},
      eprint={2404.02942},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.02942}, 
}
```
