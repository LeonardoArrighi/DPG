import os
import argparse

import fhg.test_sklearn_work as test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from fhg.core import digraph_to_nx, get_fhg, get_critical_nodes, get_fhg_node_metrics, get_fhg_metrics, critical_nodes_performance
from fhg.visualizer import basic_plot, plot_rf2fhg, plot_custom_map, plot_communities_map, paper_plot
from fhg.plots import enriched_rf_importance, importance_vs_criticalscore, criticalscores_class, importance_vs_critical

import networkx as nx
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="/datasets/Indications_bin_synth_ds.csv", help="Dataset path")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the RF")
    parser.add_argument("--pv", type=float, default=0.001, help="Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.")
    parser.add_argument("--t", type=int, default=1, help="Decimal precision of each feature")
    parser.add_argument("--dir", type=str, default="test/", help="Folder to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the FHG, add the argument to use it as True")
    # parser.add_argument("--png", action='store_true', help="Save the FHG, add the argument to use it as True")
    args = parser.parse_args()

    n_learners = args.l
    perc_var = args.pv
    decimal_threshold = args.t
    plot = args.plot
    file_name = os.path.join(args.dir, f'Indications{args.l}_pv{args.pv}_t{args.t}_stats.txt')


    num_features = 16
    feature_names =  [f'F{i}' for i in range(1, num_features + 1)]
    target_list = ["Class"]
    dt = pd.read_csv(args.ds)
    dt[target_list] = dt[target_list].apply(lambda x: pd.factorize(x)[0])
    
    
    X_train, X_test, y_train, y_test = train_test_split(dt[feature_names], dt[target_list], test_size=0.2, random_state=42)

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
    print("Extracting DPG...")
    dot = get_fhg(X_train.values, feature_names, rf_classifier, perc_var, decimal_threshold)
    
    # # FHG Extraction custom dataset
    # dot = get_fhg(X_train, features, rf_classifier, perc_var, decimal_threshold)
    
    fhg_model, nodes_list = digraph_to_nx(dot)

    if len(nodes_list) < 2:
        print("Warning: Less than two nodes resulted.")
    
    print("Calculating Graph Metrics...")
    df_fhg_metrics = get_fhg_metrics(fhg_model, nodes_list)
    
    print("Calculating Nodes Metrics...")
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
            
            "new_bl"
            + str(n_learners)
            + "_perc"
            + str(perc_var)
            + "_dec"
            + str(decimal_threshold)
        )
        paper_plot(plot_name, dot, df)
        #plot_custom_map(plot_name, dot, df, attribute='Betweness centrality', norm_flag=True, class_flag=False)          # # Metric graph
    
    df.sort_values(['Degree'])

    df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_node_metrics_(crit{length}).csv'),
                encoding='utf-8')
    # print(df[['Local reaching centrality', 'Label']].sort_values(by=['Local reaching centrality']))
    with open(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_fhg_metrics_(crit{length}).txt'), 'w') as f:
        for key, value in df_fhg_metrics.items():
            riga = f"{key}: {value}\n"
            f.write(riga)
        
    
# python3 -W ignore to_work_with_new.py --ds datasets/dt_16feat_1ksample_4class.csv --l 5 --pv 0 --t 2 --plot --dir temp
