# Decision Predicate Graph (DPG)
DPG is a model-agnostic tool to provide a global interpretation of tree-based ensemble models, addressing transparenct and explainability challenges.

The concept behind DPG is to convert a generic tree-based ensemble model for classification into a graph, where:
- Nodes represent predicates, i.e., the feature-value associations present in each node of every tree;
- Edges denote the frequency with which these predicates are satisfied during the model training phase by the samples of the dataset.

![visualization](https://github.com/LeonardoArrighi/DPG/blob/main/examples/example.png?raw=true
)
<img src="https://github.com/LeonardoArrighi/DPG/blob/main/examples/example.png?raw=true" width="256" height="256"> 