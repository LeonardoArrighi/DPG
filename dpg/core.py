import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math
import os
import numpy as np

from tqdm import tqdm
import graphviz
import networkx as nx
import hashlib
import yaml
from joblib import Parallel, delayed

from typing import List, Dict, Union
from sklearn.base import is_classifier, is_regressor

from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor)

class DPGError(Exception):
    """Base exception class for DPG-specific errors"""
    pass


class DecisionPredicateGraph:
    """
    Main class for converting tree-based ensemble models into interpretable graphs.
    
    Attributes:
        model: Trained tree ensemble model (RandomForest, AdaBoost, etc.)
        feature_names: List of feature names
        target_names: List of target class names
        perc_var: Minimum path frequency threshold (0-1)
        decimal_threshold: Rounding precision for feature values
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    def __init__(self, model, feature_names, target_names=None, config_file="config.yaml"):
        """
        Initialize DPG converter with model and configuration.
        
        Args:
            model: Tree ensemble model with estimators_ attribute
            feature_names: List of feature names
            target_names: Optional list of target class names
            config_file: Path to YAML config file
        """
        # Load configuration
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Input validation
        if not hasattr(model, 'estimators_'):
            raise DPGError("Model must be a tree-based ensemble")
        if len(feature_names) == 0:
            raise DPGError("Feature names cannot be empty")
        
        # Initialize attributes
        self.model = model
        self.feature_names = feature_names 
        self.target_names = target_names #TODO create "Class as class name"
        self.perc_var = config['dpg']['default']['perc_var']
        self.decimal_threshold = config['dpg']['default']['decimal_threshold']
        self.n_jobs = config['dpg']['default']['n_jobs'] 

    def fit(self, X_train):
        """
        Main pipeline: Extract decision paths → Build graph → Generate visualization.
        
        Args:
            X_train: Training data (n_samples, n_features)
            
        Returns:
            graphviz.Digraph: Visualizable graph object
        """
        print("\nStarting DPG extraction *****************************************")
        print("Model Class:", self.model.__class__.__name__)
        print("Model Class Module:", self.model.__class__.__module__)
        print("Model Estimators: ", len(self.model.estimators_))
        print("Model Params: ", self.model.get_params())
        print("*****************************************************************")

        # Extract decision paths (parallel or sequential)
        if self.n_jobs == 1:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
            )
        else:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble_parallel)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
            )

        # Process extracted paths
        log = [item for sublist in log for item in sublist]
        log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])

        print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')

        # Filter infrequent paths if threshold set
        if self.perc_var > 0:
            log_df = self.filter_log(log_df)

        print('Building DPG...')
        dfg = self.discover_dfg(log_df)

        print('Extracting graph...')
        return self.generate_dot(dfg)

    def tracing_ensemble(self, case_id, sample):
        """
        Extract decision path for a single sample (generator version).
        
        Args:
            case_id: Sample identifier
            sample: Feature values (1D array)
            
        Yields:
            List[str]: Path segments as [prefix, decision/prediction]
        """
        is_regressor = isinstance(self.model, (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor))
        sample = sample.reshape(-1)
        for i, tree in enumerate(self.model.estimators_):
            tree_ = tree.tree_
            node_index = 0
            prefix = f"sample{case_id}_dt{i}"
            while True:
                left = tree_.children_left[node_index]
                right = tree_.children_right[node_index]
                if left == right:
                    if is_regressor:
                        pred = round(tree_.value[node_index][0][0], 2)
                        yield [prefix, f"Pred {pred}"]
                    else:
                        pred_class = tree_.value[node_index].argmax()
                        #Using the original class name
                        if self.target_names is not None:
                            pred_class = self.target_names[pred_class]
                        yield [prefix, f"Class {pred_class}"]
                    break
                feature_index = tree_.feature[node_index]
                threshold = round(tree_.threshold[node_index], self.decimal_threshold)
                feature_name = self.feature_names[feature_index]
                sample_val = sample[feature_index]
                if sample_val <= threshold:
                    condition = f"{feature_name} <= {threshold}"
                    node_index = left
                else:
                    condition = f"{feature_name} > {threshold}"
                    node_index = right
                yield [prefix, condition]

    #for parallel processing, when using n_jobs>1
    def tracing_ensemble_parallel(self, case_id, sample):
        is_regressor = isinstance(self.model, (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor))
        sample = sample.reshape(-1)
        result = []
        for i, tree in enumerate(self.model.estimators_):
            tree_ = tree.tree_
            node_index = 0
            prefix = f"sample{case_id}_dt{i}"
            while True:
                left = tree_.children_left[node_index]
                right = tree_.children_right[node_index]
                if left == right:
                    if is_regressor:
                        pred = round(tree_.value[node_index][0][0], 2)
                        result.append([prefix, f"Pred {pred}"])
                    else:
                        pred_class = tree_.value[node_index].argmax()
                        if self.target_names is not None:
                            pred_class = self.target_names[pred_class]
                        result.append([prefix, f"Class {pred_class}"])
                    break
                feature_index = tree_.feature[node_index]
                threshold = round(tree_.threshold[node_index], self.decimal_threshold)
                feature_name = self.feature_names[feature_index]
                sample_val = sample[feature_index]
                if sample_val <= threshold:
                    condition = f"{feature_name} <= {threshold}"
                    node_index = left
                else:
                    condition = f"{feature_name} > {threshold}"
                    node_index = right
                result.append([prefix, condition])
        return result
            

    def filter_log(self, log):
        """
        Filter paths based on frequency threshold.
        
        Args:
            log: DataFrame of extracted paths
            
        Returns:
            pd.DataFrame: Filtered paths meeting perc_var threshold
        """
        from collections import defaultdict
        variant_map = defaultdict(list)
        for case_id, group in log.groupby("case:concept:name", sort=False):
            variant = "|".join(group["concept:name"].values)
            variant_map[variant].append(case_id)

        case_ids_to_keep = set()
        min_count = len(log["case:concept:name"].unique()) * self.perc_var
        for variant, case_ids in variant_map.items():
            if len(case_ids) >= min_count:
                case_ids_to_keep.update(case_ids)
        return log[log["case:concept:name"].isin(case_ids_to_keep)].copy()

    def discover_dfg(self, log):
        """
        Build directed frequency graph from path logs.
        
        Args:
            log: DataFrame of decision paths
            
        Returns:
            Dict[tuple, int]: Edge frequencies as {(source, target): count}
        """
        def process_chunk(chunk):
            """Process a subset of cases in parallel"""
            chunk_dfg = {}
            for case in tqdm(chunk, desc="Processing cases", leave=False):
                trace_df = log[log["case:concept:name"] == case].copy()
                trace_df.sort_values(by="case:concept:name", inplace=True)
                for i in range(len(trace_df) - 1):
                    key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])
                    chunk_dfg[key] = chunk_dfg.get(key, 0) + 1
            return chunk_dfg

        cases = log["case:concept:name"].unique()
        if len(cases) == 0:
            raise Exception("There is no paths with the current value of perc_var and decimal_threshold!")

        if self.n_jobs == -1:
            self.n_jobs = os.cpu_count()

        chunk_size = max(len(cases) // self.n_jobs, 1)
        chunks = [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]
        results = Parallel(n_jobs=self.n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)

        dfg = {}
        for result in results:
            for key, value in result.items():
                dfg[key] = dfg.get(key, 0) + value
        return dfg

    def generate_dot(self, dfg):
        """
        Convert frequency graph to Graphviz format.
        
        Args:
            dfg: Directed frequency graph
            
        Returns:
            graphviz.Digraph: Visualizable graph
        """
        dot = graphviz.Digraph(
            "dpg",
            engine="dot",
            graph_attr={"bgcolor": "white", "rankdir": "R", "overlap": "false", "fontsize": "20"},
            node_attr={"shape": "box"},
        )
        added_nodes = set()
        for k, v in sorted(dfg.items(), key=lambda item: item[1]):
            for activity in k:
                if activity not in added_nodes:
                    dot.node(
                        str(int(hashlib.sha1(activity.encode()).hexdigest(), 16)),
                        label=activity,
                        style="filled",
                        fontsize="20",
                        fillcolor="#ffc3c3",
                    )
                    added_nodes.add(activity)
            dot.edge(
                str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
                str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
                label=str(v),
                penwidth="1",
                fontsize="18"
            )
        return dot

    def to_networkx(self, graphviz_graph):
        """
        Convert Graphviz graph to NetworkX format.
        
        Args:
            graphviz_graph: Input graph
            
        Returns:
            Tuple[nx.DiGraph, List]: NetworkX graph and node metadata
        """
        networkx_graph = nx.DiGraph()
        nodes_list = []
        edges = []
        weights = {}
        for edge in graphviz_graph.body:
            if "->" in edge:
                src, dest = edge.split("->")
                src = src.strip()
                dest = dest.split(" [label=")[0].strip()
                weight = None
                if "[label=" in edge:
                    attr = edge.split("[label=")[1].split("]")[0].split(" ")[0]
                    weight = float(attr) if attr.replace(".", "").isdigit() else None
                    weights[(src, dest)] = weight
                edges.append((src, dest))
            if "[label=" in edge:
                id, desc = edge.split("[label=")
                id = id.replace("\t", "").replace(" ", "")
                desc = desc.split(" fillcolor=")[0].replace('"', "")
                nodes_list.append([id, desc])
        for src, dest in edges:
            if (src, dest) in weights:
                networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
            else:
                networkx_graph.add_edge(src, dest)
        return networkx_graph, sorted(nodes_list, key=lambda x: x[0])
