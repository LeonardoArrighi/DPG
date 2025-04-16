import pandas as pd
import numpy as np
import ntpath
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier, is_regressor

from .core import DecisionPredicateGraph
from .visualizer import plot_dpg


def select_custom_dataset(path, target_column):
    """
    Loads a custom dataset from a CSV file, separates the target column, and prepares the data for modeling.

    Args:
        path: The file path to the CSV dataset.
        target_column: The name of the column to be used as the target variable.

    Returns:
        data: A numpy array containing the feature data.
        features: A numpy array containing the feature names.
        target: A numpy array containing the target variable.
    """
    df = pd.read_csv(path, sep=',')

    if target_column is None:
        target_column = df.columns[-1]
        print(f"[INFO] No target column specified. Using last column: {target_column}")

    # Extract the target variable
    target = df[target_column].values

    # Drop target column from features
    df.drop(columns=[target_column], inplace=True)

    # Sanitize feature data
    df = df.apply(pd.to_numeric, errors='coerce')  # convert non-numeric to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    # Round and convert to float64
    data = np.round(df.values, 2).astype(np.float64)
    features = df.columns.values

    return data, features, target




def test_base_sklearn(datasets, target_column, n_learners, perc_var, decimal_threshold, model_name='RandomForestClassifier', file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, class_flag=False):
    """
    Trains a Random Forest classifier on a selected dataset, evaluates its performance, and optionally plots the DPG.

    Args:
    datasets: The path to the custom dataset to use.
    target_column: The name of the column to be used as the target variable.
    n_learners: The number of trees in the Random Forest.
    perc_var: Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.
    decimal_threshold: Decimal precision of each feature.
    model_name: The name of the model chosen. Default is RandomForestClassifier.
    file_name: The name of the file to save the evaluation results. If None, prints the results to the console.
    plot: Boolean indicating whether to plot the DPG. Default is False.
    save_plot_dir: Directory to save the plot image. Default is "examples/".
    attribute: A specific node attribute to visualize. Default is None.
    communities: Boolean indicating whether to visualize communities. Default is False.
    class_flag: Boolean indicating whether to highlight class nodes. Default is False.

    Returns:
    df: A pandas DataFrame containing node metrics.
    df_dpg: A pandas DataFrame containing DPG metrics.
    """
    
    
    if datasets is None:
        raise Exception("Please provide a dataset.")

    data, features, target = select_custom_dataset(datasets, target_column=target_column)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )
    
    # Train model
    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=n_learners, random_state=42)
    elif model_name == 'ExtraTreesClassifier':
        ExtraTreesClassifier(n_estimators=n_learners, random_state=42)
    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(n_estimators=n_learners, random_state=42)
    elif model_name == 'BaggingClassifier':
        BaggingClassifier(n_estimators=n_learners, random_state=42)
    else:
        raise Exception("The selected model is not currently available.")
            
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if is_classifier(model):
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Print or save the evaluation results
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(f'Statistics for the model: {model_name}\n\n')
                f.write(f'Accuracy: {accuracy:.2f}\n')
                f.write(f'F1 Score: {f1:.2f}\n')
                f.write('\nConfusion Matrix:\n')
                for i in confusion:
                    f.write(f'{str(i)}\n')
                f.write('\nClassification Report:')
                f.write(classification_rep)
        else:
            print(f'Accuracy: {accuracy:.2f}')
            print('Confusion Matrix:')
            print(confusion)
            print('Classification Report:')
            print(classification_rep)
            
    elif is_regressor(model):
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)

        # Print or save the evaluation results
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(f'Statistics for the model: {model_name}\n\n')
                f.write(f'Mean Squared Error: {mse:.2f}')
        else:
            print(f"Mean Squared Error: {mse:.2f}")


    # Extract DPG
    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=features,
        target_names=np.unique(target).astype(str).tolist(),
        perc_var=perc_var,
        decimal_threshold=decimal_threshold,
        n_jobs=1
    )
    dot = dpg.fit(X_train)
    
    # Convert Graphviz Digraph to NetworkX DiGraph
    dpg_model, nodes_list = dpg.to_networkx(dot)

    if len(nodes_list) < 2:
        print("Warning: Less than two nodes resulted.")
        return
    
    # Get metrics from the DPG
    df_dpg = dpg.extract_graph_metrics(dpg_model, nodes_list)
    df = dpg.extract_node_metrics(dpg_model, nodes_list)
    
    
    # Plot the DPG if requested
    if plot:
        plot_name = (
            os.path.splitext(ntpath.basename(datasets))[0]
            + "_"
            + model_name
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
