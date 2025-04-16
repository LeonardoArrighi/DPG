import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
from dpg.core import DecisionPredicateGraph
from dpg.visualizer import plot_dpg, plot_dpg_reg

approach = "AutoCluster"
list_datasets = ["autocluster_as.csv"]
list_perc_var = [0.0001]
num_bl = 10
model = RandomForestClassifier(n_estimators=num_bl, random_state=27)

for perc_var in list_perc_var:
    for dataset in list_datasets:
        print("bl", num_bl)
        current_path = os.getcwd()
        dataset_path = os.path.join(current_path, "datasets_poac", dataset)
        dataset_raw = pd.read_csv(dataset_path, index_col=0)

        features = dataset_raw.iloc[:, :-1]
        target_column = dataset_raw.columns[-1]
        feature_names = dataset_raw.columns[:-1]

        features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())

        print("Size of X", features.shape)
        X = np.round(features, 2)
        y = dataset_raw[target_column]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        if isinstance(model, RandomForestRegressor):
            mse_scores, mae_scores, r2_scores = [], [], []
        else:
            accuracy_scores, f1_scores = [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if isinstance(model, RandomForestRegressor):
                mse_scores.append(mean_squared_error(y_test, y_pred))
                mae_scores.append(mean_absolute_error(y_test, y_pred))
                r2_scores.append(r2_score(y_test, y_pred))
                print(f"Fold - MSE: {mse_scores[-1]}, MAE: {mae_scores[-1]}, R2: {r2_scores[-1]}")
            else:
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                print(f"Fold - Accuracy: {accuracy_scores[-1]}, F1-Score: {f1_scores[-1]}")

        if isinstance(model, RandomForestRegressor):
            mean_r2 = np.mean(r2_scores)
            metric_suffix = f"r2_{np.round(mean_r2, 2)}"
        else:
            mean_accuracy = np.mean(accuracy_scores)
            metric_suffix = f"acc_{np.round(mean_accuracy, 2)}"

        # DPG Extraction
        dpg = DecisionPredicateGraph(model=model, feature_names=feature_names, perc_var=perc_var, decimal_threshold=2, n_jobs=1)
        dot = dpg.fit(X_train.values)
        dpg_model, nodes_list = dpg.to_networkx(dot)

        # Metrics
        dpg_metrics = dpg.extract_graph_metrics(dpg_model, nodes_list, class_names=np.unique(y).astype(str).tolist())
        df = dpg.extract_node_metrics(dpg_model, nodes_list)

        # Save
        metrics_file = os.path.join(current_path, f'datasets_poac/{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_perc_{perc_var}_dpg_metrics.txt')
        with open(metrics_file, 'w') as f:
            for key, value in dpg_metrics.items():
                f.write(f"{key}: {value}\n")

        df.to_csv(os.path.join(current_path, f'datasets_poac/{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_perc_{perc_var}_node_metrics.csv'), encoding='utf-8')

        # Plot
        if isinstance(model, RandomForestRegressor):
            plot_dpg_reg(f'{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_dpg_metrics.png', dot, df, dpg_metrics, save_dir="datasets_poac/", communities=True, leaf_flag=True)
        else:
            plot_dpg(f'{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_dpg_metrics.png', dot, df, dpg_metrics, save_dir="datasets_poac/", communities=True, class_flag=True)
