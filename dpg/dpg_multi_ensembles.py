from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.base import is_classifier, is_regressor
from dpg.core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from dpg.visualizer import plot_dpg

import numpy as np
import pandas as pd

import neptune
import argparse

import os


def test_base_multi(model, datasets, n_learners, perc_var, decimal_threshold, file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, class_flag=False, run=None, perc_data=100):
    # Load dataset
    #dt = select_dataset(datasets)

    class Dataset:
        def __init__(self, data, target, feature_names):
            self.data = data
            self.target = target
            self.feature_names = feature_names

    # Read dataset
    datasets = os.path.join(datasets)
    print('>::::::::::::::::::>>>>'+datasets)
    dataset_raw = pd.read_csv(datasets)
    dataset_raw = dataset_raw.iloc[:int(dataset_raw.shape[0]*perc_data/100), :]
    print('n_samples', int(dataset_raw.shape[0]))
    run["n_samples"] = int(dataset_raw.shape[0])
    

    # Create an instance of Dataset class and assign data
    dt = Dataset(
        data=dataset_raw.iloc[:, :-1].values,         # Feature columns
        target=dataset_raw.iloc[:, -1].values,        # Target column
        feature_names=dataset_raw.columns[:-1].tolist()  # Feature names
    )
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dt.data, dt.target, test_size=0.3, random_state=42)
    
    # Train Classifier
    ensemble_classifier = model
    ensemble_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble_classifier.predict(X_test)
    
    # Evaluate the model
    if is_classifier(ensemble_classifier):
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')  # You can adjust the 'average' parameter as needed
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Print or save the evaluation results
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(f'Accuracy: {accuracy:.2f}\n')
                f.write(f'F1 Score: {f1:.2f}\n')  # Save F1 score
                f.write('\nConfusion Matrix:\n')
                for i in confusion:
                    f.write(f'{str(i)}\n')
                f.write('\nClassification Report:\n')
                f.write(classification_rep)
        
                print(f'Accuracy: {accuracy:.2f}')
                print(f'F1 Score: {f1:.2f}')  # Print F1 score
                run["f1"] = str(f1)
                run["acc"] = str(accuracy)
                print('Confusion Matrix:')
                print(confusion)
                print('Classification Report:')
                print(classification_rep)
    else:
        # If it's a regressor, calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
                

    # Extract DPG
    dot = get_dpg(X_train, dt.feature_names, ensemble_classifier, perc_var, decimal_threshold, num_classes=len(np.unique(y_train)))

    # Convert Graphviz Digraph to NetworkX DiGraph
    dpg_model, nodes_list = digraph_to_nx(dot)

    if len(nodes_list) < 2:
        print("Warning: Less than two nodes resulted.")
        return
    
    # Get metrics from the DPG
    df_dpg = get_dpg_metrics(dpg_model, nodes_list)
    df = get_dpg_node_metrics(dpg_model, nodes_list)
    
    # Plot the DPG if requested
    if plot:
        plot_name = (
            datasets
            + "_"
            + ensemble_classifier.__class__.__name__
            + "_acc"
            + str(accuracy)
            + "_f1"
            + str(f1)
            + "_bl"
            + str(n_learners)
            + "_perc"
            + str(perc_var)
            + "_dec"
            + str(decimal_threshold)
        )

        plot_dpg(
            plot_name,
            dot,
            df,
            df_dpg,
            save_dir=save_plot_dir,
            attribute=attribute,
            communities=communities,
            class_flag=class_flag
        )
    
    return df, df_dpg


def main(file):
    
    list_n_learners = [3, 5, 7, 10, 20, 100]
    perc_dataset = [0.1, 1, 10, 25, 50, 75, 100]

    for perc_data in perc_dataset:
        for n_learners in list_n_learners:
            
            list_of_ensembles = [
                ExtraTreesClassifier(n_estimators=n_learners, random_state=42),
                RandomForestClassifier(n_estimators=n_learners, random_state=42),
                AdaBoostClassifier(n_estimators=n_learners, random_state=42),
                BaggingClassifier(n_estimators=n_learners, random_state=42),
            ]
            
            for ensemble in list_of_ensembles:
                run = neptune.init_run(
                    project="sbarbonjr/DPGPOAC",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZTM0MGEzOC1mMzM1LTRkZmMtOGRlNC00NTBmZDY3ODQ4ZTgifQ==",
                )  # your credentials
                run["file"] = file    
                print(file)

                run["n_learners"] = str(n_learners)
            
                class_name = ensemble.__class__.__name__
                print(f"The class name is: {class_name}")
                run["class_name"] = str(class_name)

                # Call the test_base_multi function with the file_name as a parameter
                df, df_dpg_metrics = test_base_multi(
                    ensemble, file, n_learners, 
                    perc_var=0.001, decimal_threshold=2, file_name=file.split('.')[0]+"_statistics.txt", plot=False, 
                    save_plot_dir="dpg_poac/", attribute=None, communities=False, class_flag=False, run=run, perc_data=perc_data)

                # Save the dataframe to a CSV file with a dynamic name
                df.sort_values(['Degree'])

                df.to_csv(f"{file}_{class_name}_{n_learners}_node_metrics.csv", encoding='utf-8')

                with open(f"{file}_{class_name}_{n_learners}_dpg_metrics.txt", 'w') as f:
                    for key, value in df_dpg_metrics.items():
                        f.write(f"{key}: {value}\n")

                run.stop()
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run machine learning models and save results to a CSV file.")
    parser.add_argument('--f', type=str, help="The name of the file to be used in the process.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the file_name received from the command line
    main(args.f)