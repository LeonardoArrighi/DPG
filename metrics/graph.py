import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from typing import Dict, List, Set
import re
import math

class GraphMetrics:
    """Handles graph-level metric calculations"""
    
    def __init__(self, target_names=None):
        self.target_names = target_names

    @staticmethod
    def calculate_class_boundaries(key: str, nodes: List[str], class_names: List[str]) -> tuple:
        """Static method for boundary calculation"""
        feature_bounds = {}
        boundaries = []
        for node in nodes:
            parts = re.split(' <= | > ', node)
            if len(parts) != 2:
                continue
            feature, value_str = parts
            try:
                value = float(value_str)
            except ValueError:
                continue
                
            if feature not in feature_bounds:
                feature_bounds[feature] = [math.inf, -math.inf]
                
            if '>' in node:
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
        return str(key), boundaries

    @classmethod
    def calculate_boundaries(cls, class_dict: Dict, class_names: List[str]) -> Dict:
        """Parallel boundary calculation"""
        results = Parallel(n_jobs=-1)(
            delayed(cls.calculate_class_boundaries)(key, nodes, class_names) 
            for key, nodes in class_dict.items()
        )
        return dict(results)

    @classmethod
    def extract_graph_metrics(cls, dpg_model: nx.DiGraph, nodes_list: List[tuple], target_names: List[str]) -> Dict:
        """Main interface for graph metrics"""
        # Create node mappings
        diz_nodes = {node[1]: node[0] for node in nodes_list if not node[0].startswith('->')}
        diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}
        
        # Community detection
        communities = list(nx.community.asyn_lpa_communities(dpg_model, weight='weight'))
        communities_labels = [
            {diz_nodes_reversed[str(node)] for node in community} 
            for community in communities
        ]
        
        # Class boundaries
        terminal_nodes = {
            k: v for k, v in diz_nodes.items() 
            if any(x in k for x in ['Class', 'Pred'])
        }
        predecessors = {}
        
        for class_name, node_id in terminal_nodes.items():
            try:
                preds = nx.descendants(dpg_model.reverse(), node_id)
                predecessors[class_name] = [
                    diz_nodes[p] for p in preds 
                    if p in diz_nodes and not any(
                        x in diz_nodes[p] for x in ['Class', 'Pred']
                    )
                ]
            except nx.NetworkXError:
                predecessors[class_name] = []
        
        # Calculate boundaries
        class_bounds = cls.calculate_boundaries(predecessors, target_names)
        
        return {
            "Communities": communities_labels,
            "Class Bounds": class_bounds
        }