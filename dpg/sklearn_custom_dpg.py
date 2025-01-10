import pandas as pd
import numpy as np
import ntpath
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier, is_regressor

from .core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from .visualizer import plot_dpg
from .categorical_ensemble import DataPreprocessor


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
    # Load the dataset from the specified CSV file
    df = pd.read_csv(path, sep=',')
    
    # Extract the target variable
    target = np.array(df[target_column])
    
    # Remove the target column from the dataframe
    df.drop(columns=[target_column], inplace=True)
    
    preprocessor = None
    if has_non_numerical_features(df):
        # Transform the data
        preprocessor = DataPreprocessor(target_column)
        transformed_df = preprocessor.fit_transform(df)
        #transformed_df.to_csv('adult_mini_transformed.csv', index=False)
        df = transformed_df
    
        # Convert the feature data to a numpy array
    data = []
    for index, row in df.iterrows():
        data.append([row[j] for j in df.columns])
    data = np.array(data)
    
    # Extract feature names
    features = np.array([i for i in df.columns])

    # Return the feature data, feature names, and target variable
    return data, features, target, preprocessor



def test_base_custom(datasets, target_column, n_learners, perc_var, decimal_threshold, model_name='RandomForestClassifier', file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, class_flag=False):
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
    
    # Load dataset
    data, features, target, preprocessor  = select_custom_dataset(datasets, target_column=target_column)
    
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
    dot = get_dpg(X_train, features, model, perc_var, decimal_threshold, preprocessor)
    
    # Convert Graphviz Digraph to NetworkX DiGraph  
    dpg_model, nodes_list = digraph_to_nx(dot)

    if len(nodes_list) < 2:
        raise Exception("Warning: Less than two nodes resulted.")
        
    
    # Get metrics from the DPG
    df_dpg = get_dpg_metrics(dpg_model, nodes_list, preprocessor)
    df = get_dpg_node_metrics(dpg_model, nodes_list)
    
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

def has_non_numerical_features(data):
    """
    Checks if the given DataFrame has at least one non-numerical feature.

    Parameters:
    - data (pd.DataFrame): The DataFrame to check.

    Returns:
    - bool: True if there is at least one non-numerical feature, False otherwise.
    """
    # Select columns that are not of numeric types
    non_numeric_columns = data.select_dtypes(exclude=[np.number, 'bool']).columns
    # Check if there are any non-numeric columns
    return len(non_numeric_columns) > 0