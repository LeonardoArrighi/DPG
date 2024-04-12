from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_diabetes,
)

from .core import digraph_to_nx, get_fhg, get_critical_nodes, get_fhg_node_metrics, get_fhg_metrics, critical_nodes_performance
from .visualizer import basic_plot, plot_rf2fhg, plot_custom_map, plot_communities_map, paper_plot
from .plots import enriched_rf_importance, importance_vs_criticalscore, criticalscores_class, importance_vs_critical

import networkx as nx
import pandas as pd
# from category_encoders import OneHotEncoder
# from sklearn.compose import ColumnTransformer


def select_dataset(name):
    datasets = {
        "iris": load_iris(),
        "diabetes": load_diabetes(),
        "digits": load_digits(),
        "wine": load_wine(),
        "cancer": load_breast_cancer(),
    }

    return datasets.get(name.lower(), None)

# def select_custom_dataset(path, target_column):
#     df = pd.read_csv(path, sep=',')
#     target = df[target_column]
#     df.drop(columns=[target_column], inplace=True)
#     df_cat = df.select_dtypes('object')
#     df_categorical = df_cat.columns
#     preprocessor = ColumnTransformer(
#         transformers=[('onehot', OneHotEncoder(), df_categorical)],
#         remainder='passthrough')
#     data = preprocessor.fit_transform(df)
#     features = preprocessor.get_feature_names_out()

#     return data, features, target

def test_base_sklearn(datasets, n_learners, perc_var, decimal_threshold, plot=False, file_name=None):
    
    # Load dataset
    dt = select_dataset(datasets)

    # # Load custom dataset
    # data, features, target = select_custom_dataset(datasets, target_column = 'HeartDisease')
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        dt.data, dt.target, test_size=0.3, random_state=42
    )

    # # Split custom dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data, target, test_size=0.3, random_state=42
    # )
    
    # RF
    rf_classifier = RandomForestClassifier(n_estimators=n_learners, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Print the results
    if file_name is not None:
        f = open(file_name, "w")
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write('\nConfusion Matrix:\n')
        for i in confusion:
            f.write(f'{str(i)}\n')
        f.write('\nClassification Report:')
        f.write(classification_rep)
        f.close()
    else:
        print(f'Accuracy: {accuracy:.2f}')
        print('Confusion Matrix:')
        print(confusion)
        print('Classification Report:')
        print(classification_rep)

    # FHG Extraction
    dot = get_fhg(X_train, dt.feature_names, rf_classifier, perc_var, decimal_threshold)
    
    # # FHG Extraction custom dataset
    # dot = get_fhg(X_train, features, rf_classifier, perc_var, decimal_threshold)
    
    fhg_model, nodes_list = digraph_to_nx(dot)

    if len(nodes_list) < 2:
        print("Warning: Less than two nodes resulted.")
        return
    
    df_fhg_metrics = get_fhg_metrics(fhg_model, nodes_list)
    df = get_fhg_node_metrics(fhg_model, nodes_list)
    

    # # FINDING CRITICAL NODE
    # result = get_critical_nodes(df, fhg_model, nodes_list, len(rf_classifier.estimators_), X_train.shape[0], True)
    # if result is not None:
    #     cn_list, cn_list_items = result
    #     length = len(cn_list)
    #     # df_cn_perf = critical_nodes_performance(df,fhg_model,cn_list_items,nodes_list, pd.concat([pd.DataFrame(dt.data, columns=dt.feature_names), pd.DataFrame(dt.target, columns=['target'])], axis=1))
    # else:
    #     cn_list = None
    #     length = 0
    #     # df_cn_perf = None
    #     print('## There is no critical nodes ##')
    

    # ------------ MODIFY ---------------
    cn_list = None
    length = 0

    if plot:
        plot_name = (
            datasets
            + "_bl"
            + str(n_learners)
            + "_perc"
            + str(perc_var)
            + "_dec"
            + str(decimal_threshold)
        )

        # # CHOOSE THE PLOT
        # paper_plot(plot_name, dot, df)
        # plot_rf2fhg(plot_name, dot, cn_list)                                                                                  # # Standard graph
        plot_custom_map(plot_name, dot, df, attribute='Local reaching centrality', norm_flag=True, class_flag=False)          # # Metric graph
        # plot_custom_map(plot_name, dot, df, attribute='Betweness centrality', norm_flag=True, class_flag=False)                # # Metric graph
        # plot_communities_map(plot_name, dot, df, df_fhg_metrics['Communities'])                                               # # Communities graph

    # importance_vs_criticalscore(rf_classifier, dtail_list, dt.feature_names)
        
    # criticalscores_class(dtail_list)

    # importance_vs_critical(rf_classifier, dtail_list, pd.DataFrame(dt.data, columns=dt.feature_names), dt.feature_names)

    # enriched_rf_importance(rf_classifier, dtail_list, dt.feature_names)
    
    return df, df_fhg_metrics, length#, df_cn_perf
