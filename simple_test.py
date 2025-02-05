import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from dpg.core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from dpg.visualizer import plot_dpg_reg, plot_dpg

import networkx as nx
import os


#PoAC:
#list_datasets = ["30k_training_sv6.csv"]
#list_num_bl = [10]
#list_perc_var = [0.0001]
#model = RandomForestRegressor(n_estimators=num_bl, random_state=27)

#ML2DAC
#list_datasets = ["ml2dac_stats_info_gen.csv"]
#list_perc_var = [0.0001]
#num_bl = 15
#model = RandomForestClassifier(n_estimators=num_bl, random_state=27)

#ZAP
#list_datasets = ["zap_as_full.csv"]
#list_perc_var = [0.00001]
#num_bl = 100
#model = RandomForestClassifier(n_estimators=num_bl, random_state=27)

#approach = "ZAP"
#list_datasets = ["zap_hpo_13k.csv"]
#list_perc_var = [0.0001]
#num_bl = 10
#model = RandomForestRegressor(n_estimators=num_bl, random_state=27)

#approach = "AutoClust_HPO"
#list_datasets = ["autoclust_hpo.csv"]
#list_perc_var = [0.0001]
#num_bl = 10
#model = RandomForestRegressor(n_estimators=num_bl, random_state=27)

approach = "AutoCluster"
list_datasets = ["autocluster_as.csv"]
list_perc_var = [0.0001]
num_bl = 10
model = RandomForestClassifier(n_estimators=num_bl, random_state=27)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the else block to handle classification models
for perc_var in list_perc_var:
    for dataset in list_datasets:
        print("bl", num_bl)
        # Paths and loading data
        current_path = os.getcwd()
        dataset_path = os.path.join(current_path, "datasets_poac", dataset)

        # Load the dataset
        dataset_raw = pd.read_csv(dataset_path, index_col=0)

        #df = dataset_raw
        #columns = df.columns.tolist()
        #columns[0], columns[-1] = columns[-1], columns[0]  # Swap first and last column
        #df = df[columns]
        #df.to_csv(dataset_path, index=False)

        # Splitting features and labels
        features = dataset_raw.iloc[:, :-1]  # All columns except the last one
        target_column = dataset_raw.columns[-1]  # The name of the last column
        feature_names = dataset_raw.columns[:-1]

        features = features.replace([np.inf, -np.inf], np.nan)  # Convert infinities to NaNs
        features = features.fillna(features.mean())  # Replace NaNs with column means

        print("Size of X", features.shape)
        X = np.round(features, 2)
        y = dataset_raw[target_column]

        # Initialize cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        if isinstance(model, RandomForestRegressor):
            mse_scores, mae_scores, r2_scores = [], [], []
        else:
            accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []

        # Cross-validation loop
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train.to_csv("X_train.csv", index=False)
            y_train.to_csv("y_train.csv", index=False)

            # Model training
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            if isinstance(model, RandomForestRegressor):
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mse_scores.append(mse)
                mae_scores.append(mae)
                r2_scores.append(r2)

                print(f"Fold - MSE: {mse}, MAE: {mae}, R2: {r2}")
            else:
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                #precision = precision_score(y_test, y_pred, average='weighted')
                #recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                accuracy_scores.append(accuracy)
                #precision_scores.append(precision)
                #recall_scores.append(recall)
                f1_scores.append(f1)

                #print(f"Fold - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
                print(f"Fold - Accuracy: {accuracy}, F1-Score: {f1}")

        if isinstance(model, RandomForestRegressor):
            # Regression results
            mean_mse, std_mse = np.mean(mse_scores), np.std(mse_scores)
            mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
            mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)

            print(f"Mean MSE: {mean_mse}, Std MSE: {std_mse}")
            print(f"Mean MAE: {mean_mae}, Std MAE: {std_mae}")
            print(f"Mean R2: {mean_r2}, Std R2: {std_r2}")
        else:
            # Classification results
            mean_accuracy, std_accuracy = np.mean(accuracy_scores), np.std(accuracy_scores)
            #mean_precision, std_precision = np.mean(precision_scores), np.std(precision_scores)
            # mean_recall, std_recall = np.mean(recall_scores), np.std(recall_scores)
            mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)

            print(f"Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}")
            #print(f"Mean Precision: {mean_precision}, Std Precision: {std_precision}")
            #print(f"Mean Recall: {mean_recall}, Std Recall: {std_recall}")
            print(f"Mean F1-Score: {mean_f1}, Std F1-Score: {std_f1}")

        # Extract DPG (Assuming this is applicable for both regression and classification)
        dot = get_dpg(X_train.values, feature_names, model, perc_var, 2)

        # Convert Graphviz Digraph to NetworkX DiGraph
        dpg_model, nodes_list = digraph_to_nx(dot)

        # Get metrics from the DPG
        df_dpg = get_dpg_metrics(dpg_model, nodes_list)
        metric_suffix = f"r2_{np.round(mean_r2, 2)}" if isinstance(model, RandomForestRegressor) else f"acc_{np.round(mean_accuracy, 2)}"
        with open(os.path.join(current_path, f'datasets_poac/{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_perc_{perc_var}_dpg_metrics.txt'), 'w') as f:
            for key, value in df_dpg.items():
                f.write(f"{key}: {value}\n")

        df = get_dpg_node_metrics(dpg_model, nodes_list)
        df.to_csv(os.path.join(current_path, f'datasets_poac/{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_perc_{perc_var}_node_metrics.csv'),
                  encoding='utf-8')

        # Plot DPG
        if isinstance(model, RandomForestRegressor):
            plot_dpg_reg(f'{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_dpg_metrics.png', dot, df, df_dpg, save_dir="datasets_poac/", communities=True, leaf_flag=True)
        else:
            plot_dpg(f'{model.__class__.__name__}_{approach}_s{features.shape[0]}_bl{num_bl}_{metric_suffix}_dpg_metrics.png', dot, df, df_dpg, save_dir="datasets_poac/", communities=True, class_flag=True)
