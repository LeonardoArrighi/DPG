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
from joblib import Parallel, delayed

from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor)

class DecisionPredicateGraph:
    def __init__(self, model, feature_names, target_names=None, perc_var=0.0, decimal_threshold=2, n_jobs=-1):
        self.model = model
        self.feature_names = feature_names 
        self.target_names = target_names #TODO create "Class as class name"
        self.perc_var = perc_var
        self.decimal_threshold = decimal_threshold
        self.n_jobs = n_jobs

    def fit(self, X_train):
        print("\nStarting DPG extraction *****************************************")
        print("Model Class:", self.model.__class__.__name__)
        print("Model Class Module:", self.model.__class__.__module__)
        print("Model Estimators: ", len(self.model.estimators_))
        print("Model Params: ", self.model.get_params())
        print("*****************************************************************")


        if self.n_jobs == 1:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
            )
        else:
            log = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tracing_ensemble_parallel)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
            )

        log = [item for sublist in log for item in sublist]
        log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])

        print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')
        if self.perc_var > 0:
            log_df = self.filter_log(log_df)

        print('Building DPG...')
        dfg = self.discover_dfg(log_df)

        print('Extracting graph...')
        return self.generate_dot(dfg)

    def tracing_ensemble(self, case_id, sample):
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

        def process_chunk(chunk):
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

    def calculate_class_boundaries(self, key, nodes, class_names):
        class_name = key #TODO create "Class as class name"
        feature_bounds = {}
        boundaries = []
        for node in nodes:
            parts = re.split(' <= | > ', node)
            feature = parts[0]
            value = float(parts[1])
            condition = '>' in node
            if feature not in feature_bounds:
                feature_bounds[feature] = [math.inf, -math.inf]
            if condition:
                if value < feature_bounds[feature][0]:
                    feature_bounds[feature][0] = value
            else:
                if value > feature_bounds[feature][1]:
                    feature_bounds[feature][1] = value
        for feature, (min_greater, max_lessequal) in feature_bounds.items():
            if min_greater == math.inf:
                boundary = f"{feature} <= {max_lessequal}"
            elif max_lessequal == -math.inf:
                boundary = f"{feature} > {min_greater}"
            else:
                boundary = f"{min_greater} < {feature} <= {max_lessequal}"
            boundaries.append(boundary)
        return str(class_name), boundaries

    def calculate_boundaries(self, class_dict, class_names):
        results = Parallel(n_jobs=-1)(
            delayed(self.calculate_class_boundaries)(key, nodes, class_names) for key, nodes in class_dict.items()
        )
        return dict(results)

    def extract_graph_metrics(self, dpg_model, nodes_list):
        np.random.seed(42)
        diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
        diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
        diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}
        asyn_lpa_communities = nx.community.asyn_lpa_communities(dpg_model, weight='weight')
        asyn_lpa_communities_stack = [{diz_nodes_reversed[str(node)] for node in community} for community in asyn_lpa_communities]
        filtered_nodes = {k: v for k, v in diz_nodes.items() if 'Class' in k or 'Pred' in k}
        predecessors = {k: [] for k in filtered_nodes}
        for key_1, value_1 in filtered_nodes.items():
            try:
                preds = nx.single_source_shortest_path(dpg_model.reverse(), value_1)
                predecessors[key_1] = [k for k, v in diz_nodes.items() if v in preds and k != key_1]
            except nx.NetworkXNoPath:
                continue
        class_bounds = self.calculate_boundaries(predecessors, self.target_names)
        return {"Communities": asyn_lpa_communities_stack, "Class Bounds": class_bounds}

    def extract_node_metrics(self, dpg_model, nodes_list):
        in_nodes = {}
        out_nodes = {}
        degree = {}
        for node in dpg_model.nodes():
            in_nodes[node] = dpg_model.in_degree(node)
            out_nodes[node] = dpg_model.out_degree(node)
            degree[node] = in_nodes[node] + out_nodes[node]
        sample_size = int(1 * len(dpg_model.nodes()))
        betweenness_centrality = nx.betweenness_centrality(dpg_model, k=sample_size, normalized=True, weight='weight', endpoints=False)
        local_reaching_centrality = {node: nx.local_reaching_centrality(dpg_model, node, weight='weight') for node in dpg_model.nodes()}
        data_node = {
            "Node": list(dpg_model.nodes()),
            "Degree": list(degree.values()),
            "In degree nodes": list(in_nodes.values()),
            "Out degree nodes": list(out_nodes.values()),
            "Betweenness centrality": list(betweenness_centrality.values()),
            "Local reaching centrality": list(local_reaching_centrality.values()),
        }
        df_data_node = pd.DataFrame(data_node).set_index('Node')
        df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index('Node')
        return pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()
