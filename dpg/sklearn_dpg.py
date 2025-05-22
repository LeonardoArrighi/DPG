import pandas as pd
import numpy as np
import ntpath
import os
from typing import Optional, Tuple, Union
import yaml

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,GradientBoostingClassifier, BaggingClassifier,ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor)
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, f1_score, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.datasets import (load_iris, load_digits, load_wine,
                             load_breast_cancer, load_diabetes)
from sklearn.base import is_classifier, is_regressor

from .core import DecisionPredicateGraph
from .visualizer import plot_dpg
from metrics.nodes import NodeMetrics
from metrics.graph import GraphMetrics


def select_dataset(source: str, target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects either a standard sklearn dataset or loads a custom CSV dataset.
    
    Args:
        source: Either a name of standard dataset or path to CSV file
        target_column: Required for custom datasets, column name for target variable
        
    Returns:
        data: Feature data array
        features: Feature names array
        target: Target variable array
    """
    # Standard sklearn datasets
    std_datasets = {
        "iris": load_iris(),
        "diabetes": load_diabetes(),
        "digits": load_digits(),
        "wine": load_wine(),
        "cancer": load_breast_cancer(),
    }
    
    if source in std_datasets:
        dataset = std_datasets[source]
        return dataset.data, dataset.feature_names, dataset.target
    
    # Custom dataset from CSV
    try:
        df = pd.read_csv(source, sep=',')
        
        if target_column is None:
            target_column = df.columns[-1]
            print(f"[INFO] Using last column as target: {target_column}")
        elif target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
            
        target = df[target_column].values
        df.drop(columns=[target_column], inplace=True)
        
        # Clean data
        df = df.apply(pd.to_numeric, errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
        
        data = np.round(df.values, 2).astype(np.float64)
        features = df.columns.values
        
        return data, features, target
        
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")


def test_dpg(datasets: str,
             target_column: Optional[str] = None,
             n_learners: int = 5,
             perc_var: float = 0.001,
             decimal_threshold: int = 2,
             n_jobs: int = -1,
             model_name: str = 'RandomForestClassifier',
             file_name: Optional[str] = None,
             plot: bool = False,
             save_plot_dir: str = "examples/",
             attribute: Optional[str] = None,
             communities: bool = False,
             class_flag: bool = False) -> Tuple[pd.DataFrame, dict]:
    
    """
    Unified function to train models and extract DPG for both standard and custom datasets.
    
    Args:
        dataset_source: Name of standard dataset or path to CSV file
        target_column: Required for custom datasets, name of target column
        ... [other parameters same as before]
        
    Returns:
        Node metrics DataFrame and DPG metrics dictionary
    """

    # Input validation
    if n_learners <= 0:
        raise ValueError("Number of learners must be positive")
    
    # Load data
    data, features, target = select_dataset(datasets, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=42
    )
    
    # Initialize model
    model_classes = {
        'RandomForestClassifier': RandomForestClassifier,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'BaggingClassifier': BaggingClassifier
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_classes.keys())}")
    
    model = model_classes[model_name](
        n_estimators=n_learners,
        random_state=42,
        n_jobs=n_jobs
    )
    
    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation output
    if file_name:
        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
    
    if is_classifier(model):
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        if file_name:
            with open(file_name, "w") as f:
                f.write(f"Model: {model_name}\nAccuracy: {metrics['accuracy']:.2f}\n")
                f.write(f"F1: {metrics['f1']:.2f}\nConfusion Matrix:\n")
                np.savetxt(f, metrics['confusion_matrix'], fmt='%d')
                f.write(f"\nClassification Report:\n{metrics['classification_report']}")
    else:
        metrics = {'mse': mean_squared_error(y_test, y_pred)}
        if file_name:
            with open(file_name, "w") as f:
                f.write(f"Model: {model_name}\nMSE: {metrics['mse']:.2f}\n")
    
    # DPG extraction
    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=features,
        target_names=np.unique(target).astype(str).tolist()
    )
    dot = dpg.fit(X_train)
    
    # Convert to NetworkX and get metrics
    dpg_model, nodes_list = dpg.to_networkx(dot)
    if len(nodes_list) < 2:
        print("Warning: Insufficient nodes for DPG analysis")
        return None, None
    

    df = NodeMetrics.extract_node_metrics(dpg_model, nodes_list)
    df_dpg = GraphMetrics.extract_graph_metrics(dpg_model, nodes_list,target_names=np.unique(y_train).astype(str).tolist())
    
    # Plot if requested
    if plot:
        os.makedirs(save_plot_dir, exist_ok=True)
        plot_name = (
            os.path.splitext(ntpath.basename(datasets))[0] 
            if '.' in datasets else datasets
        )
        plot_name += f"_{model_name}_l{n_learners}_pv{perc_var}_t{decimal_threshold}"
        
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