import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from dpg.core import DecisionPredicateGraph
from dpg.visualizer import plot_dpg

# Configurações
dataset = "custom.csv"
perc_var = 0.0001
num_bl = 10
approach = "CustomDPG"
model = RandomForestClassifier(n_estimators=num_bl, random_state=27)

# Carregamento do dataset
current_path = os.getcwd()
dataset_path = os.path.join(current_path, "datasets", dataset)
dataset_raw = pd.read_csv(dataset_path, index_col=0)

features = dataset_raw.iloc[:, :-1]
target_column = dataset_raw.columns[-1]
feature_names = dataset_raw.columns[:-1]
y = dataset_raw[target_column]

# Tratamento de dados
features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
X = np.round(features, 2)

print("Size of X", X.shape)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores, f1_scores = [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)

    print(f"Fold - Accuracy: {accuracy}, F1-Score: {f1}")

# Métricas médias
mean_accuracy = np.mean(accuracy_scores)
metric_suffix = f"acc_{np.round(mean_accuracy, 2)}"

# Extração do DPG
dpg = DecisionPredicateGraph(model=model, feature_names=feature_names, target_names=np.unique(y_train).astype(str).tolist(),  perc_var=perc_var, decimal_threshold=2, n_jobs=1)
dot = dpg.fit(X_train.values)
dpg_model, nodes_list = dpg.to_networkx(dot)

# Extração de métricas
dpg_metrics = dpg.extract_graph_metrics(dpg_model, nodes_list)
df = dpg.extract_node_metrics(dpg_model, nodes_list)

# Salvando métricas
metrics_file = os.path.join(current_path, f'datasets/{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_perc_{perc_var}_dpg_metrics.txt')
with open(metrics_file, 'w') as f:
    for key, value in dpg_metrics.items():
        f.write(f"{key}: {value}\n")

df.to_csv(os.path.join(current_path, f'datasets/{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_perc_{perc_var}_node_metrics.csv'), encoding='utf-8')

# Plotagem
plot_dpg(
    f'{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_dpg_metrics.png',
    dot,
    df,
    dpg_metrics,
    save_dir="datasets/",
    communities=True,
    class_flag=True
)
