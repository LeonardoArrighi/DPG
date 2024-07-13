# Decision Predicate Graph (DPG)
DPG is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparenct and explainability challenges.
<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/custom_l2.jpg?raw=true"/>
</p>

----------
The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

<p align="center">
  <img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/example.png?raw=true" width="400" />
</p>
