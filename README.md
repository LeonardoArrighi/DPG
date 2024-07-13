# Decision Predicate Graph (DPG)
DPG is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparency and explainability challenges.
DPG is a graph structure that captures the tree-based ensemble model and learned dataset details, preserving the relations among features, logical decisions, and predictions towards emphasising insightful points.
DPG enables graph-based evaluations and the identification of model decisions towards facilitating comparisons between features and their associated values while offering insights into the entire model.
DPG provides descriptive metrics that enhance the understanding of the decisions inherent in the model, offering valuable insights.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/custom_l2.jpg?raw=true" width="600" />
</p>

----------
## The structure
The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/example.png?raw=true" width="400" />
</p>

----------
## Metrics
The graph-based nature of DPG provides significant enhancements in the direction of a complete mapping of the ensemble structure.
| Property     | Definition | Utility |
|--------------|------------|---------|
| Constraints  | The intervals of values for each feature obtained from all predicates connected by a path that culminates in a given class. | Calculate the classification boundary values of each feature associated with each class. |
|Betweenness Centrality | Quantifies the fraction of all the shortest paths between every pair of nodes of the graph passing through the considered node. | Identify potential bottleneck nodes that correspond to crucial decisions. |
| Local Reaching Centrality | Quantifies the proportion of other nodes reachable from the local node through its outgoing edges. | Assess the importance of nodes similarly to feature importance, but enrich the information by encompassing the values associated with features across all decisions. |
| Community | A subset of nodes of the DPG which is characterised by dense interconnections between its elements and sparse connections with the other nodes of the DPG that do not belong to the community. | Understanding the characteristics of nodes to be assigned to a particular community class, identifying predominant predicates, and those that play a marginal role in the classification process. |
