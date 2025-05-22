import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple

class NodeMetrics:
    """Handles node-level metric calculations"""
    
    def extract_node_metrics(dpg_model, nodes_list):
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