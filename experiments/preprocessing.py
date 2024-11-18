from sklearn.preprocessing import KBinsDiscretizer
import neptune
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dpg.core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from dpg.visualizer import plot_dpg
import time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import argparse

def discretize_target(path, target_column, n_bins=10, encode='ordinal', strategy='quantile'):
    """
    Check if the target column is continuous. If it is, discretize it using KBinsDiscretizer.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the target column.
    - target_column (str): The name of the target column to check and discretize if continuous.
    - n_bins (int): Number of bins to use if discretizing.
    - encode (str): Encoding method for discretizer; e.g., 'ordinal'.
    - strategy (str): Strategy for binning; e.g., 'uniform', 'quantile', 'kmeans'.
    
    Returns:
    - pd.DataFrame: Modified DataFrame with discretized target column if it was continuous.
    """
    try:
        print("Discretizing...")
        df = pd.read_csv(path)
    except Exception as e:
        raise e
    
    # Check if the target column is numeric (continuous)
    if pd.api.types.is_numeric_dtype(df[target_column]):
        print(f"The target column '{target_column}' is continuous. Discretizing into {n_bins} bins...")
        plt.figure(figsize=(12, 6))

        # Plot original y_train
        plt.subplot(1, 2, 1)
        plt.hist(df[target_column], bins=n_bins, color='skyblue', edgecolor='black')
        plt.title(f"Original {target_column}")
        
        # Apply KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        binned_column_name = f'{target_column}_binned'
        df[binned_column_name] = discretizer.fit_transform(df[[target_column]])
        
        # Plot transformed y_train_binned
        plt.subplot(1, 2, 2)
        plt.hist(df[binned_column_name], bins=n_bins, color='salmon', edgecolor='black')
        plt.title(f'Discretized {binned_column_name}')

        # Adjust the layout to avoid overlap
        plt.tight_layout()

        # This is just to save the discretized file
        directory, filename = os.path.split(path)
        basename, extension = os.path.splitext(filename)
        new_filename = f"{basename}_discretized{extension}"
        new_path = os.path.join(directory, new_filename)
        
        # Save the plot as a PNG file
        plot_path = f"plots/{basename}.png"  # Specify the file path for saving the plot
        plt.savefig(plot_path, format='png')
        
        df.drop(columns=[target_column], inplace=True)
    else:
        print(f"The target column '{target_column}' is already discrete or non-numeric. No discretization applied.")
    
    return df

def _init_neptune():
    """
    Initializes a Neptune run for experiment tracking.

    This function loads the Neptune API token and project name from environment variables
    stored in a `.env` file, ensuring that sensitive information is not hard-coded into the script.

    Environment Variables Required:
    - NEPTUNE_API_TOKEN: The API token for authenticating with the Neptune service.
    - NEPTUNE_PROJECT_NAME: The project name in Neptune, in the format "workspace/project".

    Returns:
    - neptune.run: An initialized Neptune run object that can be used for logging.
    """
    
    # Load .env file to get the API token
    load_dotenv()
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    project_name = os.getenv("NEPTUNE_PROJECT_NAME")
    # Initialize Neptune run
    return neptune.init_run(
        project=project_name,
        api_token=api_token,
    )

# Function to evaluate samples sizes and log to Neptune
def evaluate_samples(path, target_column, sample_sizes=[0.1,0.2,0.4,0.6,0.8], classifiers=None):
    """
    Evaluate multiple sample sizes and classifiers, logging results to Neptune.
    
    Parameters:
    - df: discretized/classification datasets
    - sample_sizes (list): List of fractions to sample from training data. Defaults to [0.1, 0.2, 0.4, 0.8, 1.0].
    - classifiers (list): List of classifier instances to test.
    - metrics (dict): Dictionary of metric names to metric functions. Defaults to accuracy and F1.

    Returns:
    - results (dict): Nested dictionary with sample sizes, classifier names, and computed metrics.
    """
    try:
        print("Sampling...")
        df = pd.read_csv(path)
    except Exception as e:
        raise e
    
    metrics = {
        "accuracy": accuracy_score,
        "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')
    }
    
    # Prepare the data for training and testing
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True) #holdout 0.3

    results = {}  # To store results

    # Loop over each sample size
    for size in sample_sizes:
        print(f"\nSample size: {size}")
        results[size] = {}

        # Sample the training data
        X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train, y_train, train_size=size, random_state=0)
        
        # Evaluate each classifier for each sampling size
        for clf in classifiers:
            # Initialize Neptune run
            run = _init_neptune()
            clf_name = clf.__class__.__name__ 
            print(f"Training {clf_name}")
            clf.fit(X_train_sampled, y_train_sampled)
            _, filename = os.path.split(path)
            joblib.dump(clf, f"models/{filename}-{clf_name}-{size}_jlib")
            y_pred = clf.predict(X_test)
            run['name'] = filename
            run['classifier'] = clf_name
            run['sample'] = len(df)*size
            run['sample_size'] = size

            # Store metrics for each classifier and sample size
            clf_results = {metric_name: metric_func(y_test, y_pred) for metric_name, metric_func in metrics.items()}
            results[size][clf_name] = clf_results

            # Print results for this classifier
            for metric_name, score in clf_results.items():
                print(f"{metric_name.capitalize()}: {score}")
                run[f"{metric_name}"] = score
            print("\n")    
            run[f"parameters"] = str(clf.get_params())
            # Stop Neptune run
            run.stop()
    
    return results



# TODO - feature engineering
# - re-binarize groupn_data -> run preprocessing with ari_discretized as the target
# configure units vpn
# run experiment on vm
# arrumar plot, eixo x
# fork dpg 
# pull request experiment
# >> create experiment 
# adapt dpg to execute other ensembles
# eda of result -> get the graph path from 10 poac pipelines

if __name__ == "__main__":
    print(">>> Experiment DPG for Automl")
    parser = argparse.ArgumentParser(description="Experiment DPG for Automl")
    parser.add_argument("function", choices=["discretize", "sample", "dpg"], help="Choose the function to execute.")
    parser.add_argument("--path", default="datasets/",help="Path to the dataset CSV file.")
    parser.add_argument("--target_column", required=True, help="Target column in the dataset.")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for discretization.")
    parser.add_argument("--sample_sizes", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.8], help="Sample sizes to evaluate.")

    args = parser.parse_args()

    if args.function == "discretize":
        # Run the discretize function and save the result
        df = discretize_target(args.path, args.target_column, args.n_bins)
        new_path = os.path.join(os.path.dirname(args.path), f"{os.path.splitext(os.path.basename(args.path))[0]}_discretized.csv")
        df.to_csv(new_path, index=False)
        print(f"Discretized dataset saved at {new_path}")

    elif args.function == "sample":
        # Define classifiers
        list_of_ensembles = [
            ExtraTreesClassifier(n_estimators=10, random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42),
            AdaBoostClassifier(n_estimators=10, random_state=42),
            BaggingClassifier(n_estimators=10, random_state=42),
        ]
        
        list_of_datasets = ["datasets/group1_data_discretized.csv","datasets/group2_data_discretized.csv","datasets/group3_data_discretized.csv","datasets/training_sv6_discretized.csv"]
        for dataset in list_of_datasets:
            evaluate_samples(
                path=dataset,
                target_column=args.target_column,
                sample_sizes=args.sample_sizes,
                classifiers=list_of_ensembles
            )
    elif args.function == "dpg":
        run = _init_neptune()

        dataset_name = "group1_bi.csv"
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        dataset_raw = pd.read_csv(f"datasets/{dataset_name}")
        dataset_raw = dataset_raw.sample(frac=0.01, random_state=42)

        class Dataset:
            def __init__(self, data, target, feature_names):
                self.data = data
                self.target = target
                self.feature_names = feature_names

        dt = Dataset(
            data=dataset_raw.iloc[:, :-1].values,         
            target=dataset_raw.iloc[:, -1].values,        
            feature_names=dataset_raw.columns[:-1].tolist()  
        )
        run['dataset'] = dataset_name
 
        # Measure and log the training time
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(dt.data, dt.target, train_size=0.01, random_state=42, shuffle=True)
        ensemble_classifier = clf
        ensemble_classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        run['training_time'] = training_time

        perc_var = 0.001
        decimal_threshold = 2
        plot = True
        
        # Measure and log the time for get_dpg
        start_time = time.time()
        dot = get_dpg(X_train, dt.feature_names, ensemble_classifier, perc_var, decimal_threshold, num_classes=len(np.unique(y_train)))
        dpg_time = time.time() - start_time
        run['dpg_time'] = dpg_time

        # Measure and log the time for converting to NetworkX DiGraph
        start_time = time.time()
        dpg_model, nodes_list = digraph_to_nx(dot)
        nx_conversion_time = time.time() - start_time
        run['nx_conversion_time'] = nx_conversion_time

        if len(nodes_list) < 2:
            print("Warning: Less than two nodes resulted.")
            exit()

        # Measure and log the time for get_dpg_metrics
        start_time = time.time()
        df_dpg = get_dpg_metrics(dpg_model, nodes_list)
        dpg_metrics_time = time.time() - start_time
        run['dpg_metrics_time'] = dpg_metrics_time

        # Measure and log the time for get_dpg_node_metrics
        start_time = time.time()
        df = get_dpg_node_metrics(dpg_model, nodes_list)
        node_metrics_time = time.time() - start_time
        run['node_metrics_time'] = node_metrics_time

        # Plot the DPG if requested
        if plot:
            plot_name = (
                dataset_name
                + "_"
                + clf.__class__.__name__
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
                save_dir="plots/",
                attribute=None,
                communities=False,
                class_flag=False
            )
        run.stop()
        