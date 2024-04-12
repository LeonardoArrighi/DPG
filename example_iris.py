from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import fhg.core as fhg
import fhg.visualizer as vis
import fhg.plots as p


def test_iris(n_learners, perc_var, decimal_threshold=1, plot=True):
  dt = load_iris()
  feature_names = dt.feature_names
  df_features = pd.DataFrame(dt.data, columns=dt.feature_names)
  df_targets = pd.DataFrame(dt.target, columns=['target'])
  dataset = pd.concat([df_features, df_targets], axis=1)

  # Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(dt.data, dt.target, test_size=0.3, random_state=42)

  # Initialize the RandomForestClassifier
  rf_classifier = RandomForestClassifier(n_estimators=n_learners, random_state=42)

  # Fit the model to the training data
  rf_classifier.fit(X_train, y_train)

  # Make predictions on the test data
  y_pred = rf_classifier.predict(X_test)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  confusion = confusion_matrix(y_test, y_pred)
  classification_rep = classification_report(y_test, y_pred)

  # Print the results
  #print(f'Accuracy: {accuracy:.2f}')
  #print('Confusion Matrix:')
  #print(confusion)
  #print('Classification Report:')
  #print(classification_rep)

  # FHG Extraction
  dot = fhg.get_fhg(X_train, dt.feature_names, rf_classifier, perc_var, decimal_threshold)

  fhg_model, nodes_list = fhg.digraph_to_nx(dot)

  if len(nodes_list)<2:
    print("Warning: Less than two nodes resulted.")
    return
  
  df = fhg.get_fhg_metrics(fhg_model, nodes_list)
  df = fhg.get_fhg_node_metrics(fhg_model, nodes_list)
  

  # FINDING Critical Nodes
  
  cn_list, cn_list_items = fhg.get_critical_nodes(df, fhg_model, nodes_list, len(rf_classifier.estimators_), X_train.shape[0],  False)
  
  

  if plot:
    plot_name = "iris_bl"+str(n_learners)+"_perc"+str(perc_var)+"_dec"+str(decimal_threshold)
    vis.basic_plot(plot_name, dot, df)
    #vis.plot_rf2fhg(plot_name, dot, cn_list)

  #df_perf = fhg.critical_nodes_performance(df,fhg_model,cn_list_items,nodes_list, pd.concat([df_features, df_targets], axis=1))
  #df_perf.to_csv("df_perf.csv")
  #p.importance_vs_critical(rf_classifier, cn_list, dataset, feature_names)

  #p.importance_vs_criticalscore(rf_classifier, cn_list, feature_names)

  #p.enriched_rf_importance(rf_classifier, cn_list, feature_names)
  
  #p.criticalscores_class(cn_list)

  return df

n_learners = 2 #base leaners for the RF
perc_var = 0 #perc_var to 0.1, it means you want to keep only the paths that occur in at least 10% of the trees
decimal_threshold=2 #decimal precision of each feature, a small number means aggregate more nodes
plot=True #to plot the FHG

df = test_iris(n_learners,perc_var,decimal_threshold, plot)




