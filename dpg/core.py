import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math

import numpy as np

import graphviz
from itertools import combinations
import networkx as nx
from networkx.algorithms import approximation as approx # just for local_node_connectivity

import hashlib


np.random.seed(42)



def digraph_to_nx(graphviz_graph):
    '''
    This function converts a Graphviz directed graph (DiGraph) to a NetworkX directed graph (DiGraph).
    It also extracts node descriptions and edge weights from the Graphviz graph.

    Args:
    graphviz_graph: The input Graphviz directed graph.

    Returns:
    networkx_graph: The converted NetworkX directed graph.
    nodes_list: A sorted list of nodes with their descriptions.
    '''
    
    # Create an empty directed graph in NetworkX
    networkx_graph = nx.DiGraph()
    
    # Initialize a list to store nodes and a list to store edges with their weights
    nodes_list = []
    edges = []
    weights = {}

    # Extract nodes and edges from the graphviz graph
    for edge in graphviz_graph.body:
        # Check if the line represents an edge (contains '->')
        if "->" in edge:
            # Extract source and destination nodes
            src, dest = edge.split("->")
            src = src.strip()
            dest = dest.split(" [label=")[0].strip()

            # Initialize weight to None
            weight = None
            
            # Extract weight from edge attributes if available
            if "[label=" in edge:
                attr = edge.split("[label=")[1].split("]")[0].split(" ")[0]
                weight = (
                    float(attr)
                    if attr.isdigit() or attr.replace(".", "").isdigit()
                    else None
                )
                weights[(src, dest)] = weight  # Store weight for the edge

            # Add the edge to the list
            edges.append((src, dest))

        # Check if the line represents a node with attributes (contains '[label=')
        if "[label=" in edge:
            id, desc = edge.split("[label=")
            id = id.replace("\t", "")  # Clean up the node ID
            id = id.replace(" ", "")
            desc = desc.split(" fillcolor=")[0]  # Extract node description
            desc = desc.replace('"', "")
            nodes_list.append([id, desc])  # Add node to the list

    # Sort edges and nodes
    edges = sorted(edges)
    nodes_list = sorted(nodes_list, key=lambda x: x[0])

    # Add nodes and edges to the NetworkX graph
    for edge in edges:
        src, dest = edge
        # Add edge with weight if available, else add without weight
        if (src, dest) in weights:
            networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
        else:
            networkx_graph.add_edge(src, dest)

    # Return the constructed NetworkX graph and the list of nodes
    return networkx_graph, nodes_list



def tracing_rf(case_id, sample, rf_classifier, feature_names, decimal_threshold=1):
    '''
    This function traces the decision paths taken by each decision tree in a random forest classifier for a given sample.
    It records the path of decisions made by each tree, including the comparisons at each node and the resulting class.

    Args:
    case_id: An identifier for the sample being traced.
    sample: The input sample for which the decision paths are traced.
    rf_classifier: The random forest classifier containing the decision trees.
    feature_names: The names of the features used in the decision trees.
    decimal_threshold: The number of decimal places to which thresholds are rounded (default is 1).

    Returns:
    event_log: A list of the decision steps taken by each tree in the forest for the given sample.
    '''

    # Initialize an empty event log to store the decision paths
    event_log = []

    # Helper function to build the decision path for a single tree
    def build_path(tree, node_index, path=[]):
        tree_ = tree.tree_
        
        # Check if the node is a leaf node
        if tree_.children_left[node_index] == tree_.children_right[node_index]:
            path.append(f"Class {tree_.value[node_index].argmax()}")
        else:
            # Get the feature name and threshold for the current node
            feature_name = feature_names[tree_.feature[node_index]]
            threshold = round(float(tree_.threshold[node_index]), decimal_threshold)
            sample_val = sample[tree_.feature[node_index]]

            # Decide whether to go to the left or right child node based on the sample value
            if sample_val <= threshold:
                path.append(f"{feature_name} <= {round(float(tree_.threshold[node_index]), decimal_threshold)}")
                build_path(tree, tree_.children_left[node_index], path)
            else:
                path.append(f"{feature_name} > {round(float(tree_.threshold[node_index]), decimal_threshold)}")
                build_path(tree, tree_.children_right[node_index], path)

    # Loop through each decision tree in the random forest
    for i, tree_in_forest in enumerate(rf_classifier.estimators_):
        sample_path = []  # Initialize a path list for the current tree
        build_path(tree_in_forest, 0, sample_path)  # Build the path starting from the root node
        
        # Log the steps in the event log with the corresponding sample and tree identifiers
        for step in sample_path:
            event_log.append(["sample" + str(case_id) + "_dt" + str(i), step])

    # Return the event log containing the decision paths
    return event_log



def filter_log(log, perc_var):
    """
    Filters a log based on the variant percentage. Variants (unique sequences of activities for cases) 
    that occur less than the specified threshold are removed from the log.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.
    perc_var: A float representing the minimum percentage of total traces a variant must have to be kept.

    Returns:
    log: A filtered pandas DataFrame containing only the cases and activities that meet the variant percentage threshold.
    """

    # Initialize a dictionary to store variants and their corresponding cases
    variants = {}

    # Identify and count the unique variants in the log
    for case in log["case:concept:name"].unique():
        key = "|".join([x for x in log[log["case:concept:name"] == case]["concept:name"]])
        if key in variants:
            variants[key].append(case)
        else:
            variants[key] = [case]

    # Get the total number of unique traces in the log
    total_traces = log["case:concept:name"].nunique()

    # Initialize lists to store cases and activities that meet the threshold
    cases, activities = [], []

    # Filter variants based on the percentage threshold
    for k, v in variants.items():
        if len(v) / total_traces >= perc_var:
            # Add cases and their activities to the lists if they meet the threshold
            for case in v:
                for act in k.split("|"):
                    cases.append(case)
                    activities.append(act)

    # Create a new DataFrame from the filtered cases and activities
    log = pd.DataFrame(zip(cases, activities), columns=["case:concept:name", "concept:name"])

    return log



def discover_dfg(log):
    """
    Mines the nodes and edges relationships from an event log and returns a dictionary representing
    the Data Flow Graph (DFG). The DFG shows the frequency of transitions between activities.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.

    Returns:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.
    """
    # Initialize an empty dictionary to store the Data Flow Graph (DFG)
    dfg = {}
    
    # Iterate over each unique case in the log
    for case in log["case:concept:name"].unique():
        # Extract the trace (sequence of activities) for the current case
        trace_df = log[log["case:concept:name"] == case].copy()
        trace_df.sort_values(by="case:concept:name", inplace=True)
        
        # Iterate through the trace to capture transitions between consecutive activities
        for i in range(len(trace_df) - 1):
            key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])  # Create a key for the transition
            if key in dfg:
                dfg[key] += 1  # Increment the count if the transition already exists in the DFG
            else:
                dfg[key] = 1  # Initialize the count if the transition is new

    # Return the DFG dictionary
    return dfg



def generate_dot(dfg, log):
    """
    Creates a Graphviz directed graph (digraph) from a Data Flow Graph (DFG) dictionary and returns the dot representation.

    Args:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.

    Returns:
    dot: A Graphviz dot object representing the directed graph.
    """

    # Initialize a Graphviz digraph with specified attributes
    dot = graphviz.Digraph(
        "dpg",
        engine="dot",
        graph_attr={
            "bgcolor": "white",
            "rankdir": "R",
            "overlap": "false",
            "fontsize": "20",
        },
        node_attr={"shape": "box"},
    )

    # Keep track of added nodes to avoid duplicates
    added_nodes = set()
    
    # Sort the DFG dictionary by values (transition counts) for deterministic order
    sorted_dict_values = {k: v for k, v in sorted(dfg.items(), key=lambda item: item[1])}

    # Iterate through the sorted DFG dictionary
    for k, v in sorted_dict_values.items():
        
        # Add the source node to the graph if not already added
        if k[0] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
                label=f"{k[0]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[0])
        
        # Add the destination node to the graph if not already added
        if k[1] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
                label=f"{k[1]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[1])
        
        # Add an edge between the source and destination nodes with the transition count as the label
        dot.edge(
            str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
            str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
            label=str(v),
            penwidth="1",
            fontsize="18"
        )
    
    # Return the Graphviz dot object
    return dot



def calculate_boundaries(dict):
    """
    Calculates the boundaries of every feature for every class based on the provided dictionary of predecessors.

    Args:
    dict: A dictionary where keys are class labels and values are lists of predecessor node labels.

    Returns:
    boundaries_class: A dictionary containing the boundaries for each feature of every class.
    """
    # Initialize an empty dictionary to store the boundaries for each class
    boundaries_class = {}

    # Iterate over each class and its corresponding predecessor nodes
    for key, value in dict.items():
        if 'Class' in key:
            boundaries_class[key] = []
            
            # Extract unique feature names from the predecessor nodes
            key_set = []
            for i in dict[key]:
                key_set.append(str(re.split(' <= | > ', i)[0]))
            key_set = set(key_set)
            
            # Determine boundaries for each unique feature
            for valore_unico in key_set:
                match_list = [math.inf, -math.inf]
                for nodo in dict[key]:
                    if str(re.split(' <= | > ', nodo)[0]) == valore_unico:
                        if '>' in nodo:
                            if float(re.split(' > ', nodo)[1]) < match_list[0]:
                                match_list[0] = float(re.split(' > ', nodo)[1])
                        else:
                            if float(re.split(' <= ', nodo)[1]) > match_list[1]:
                                match_list[1] = float(re.split(' <= ', nodo)[1])

                # Save the boundaries as a string
                alfa = None
                if match_list[0] == math.inf:
                    alfa = str(valore_unico + " <= " + str(match_list[1]))
                elif match_list[1] == -math.inf:
                    alfa = str(valore_unico + " > " + str(match_list[0]))
                else:
                    alfa = str(str(match_list[0]) + ' < ' + valore_unico + ' <= ' + str(match_list[1]))
                boundaries_class[key].append(alfa)

    # Return the dictionary containing class boundaries
    return boundaries_class



def get_dpg_metrics(dpg_model, nodes_list):
    """
    Extracts metrics from a DPG.

    Args:
    dpg_model: A NetworkX graph representing the directed process graph.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    data: A dictionary containing the communities and class bounds extracted from the DPG model.
    """
    
    # Create a dictionary to map node labels to their identifiers
    diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
    # Remove any None keys from the dictionary
    diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
    # Create a reversed dictionary to map node identifiers to their labels
    diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}
    
    # Extract asynchronous label propagation communities
    asyn_lpa_communities = nx.community.asyn_lpa_communities(dpg_model, weight='weight')
    asyn_lpa_communities_stack = []
    for sets in asyn_lpa_communities:
        new_sets = set()
        for node in sets:
            new_sets.add(diz_nodes_reversed[str(node)])  # Map node identifiers back to labels
        asyn_lpa_communities_stack.append(new_sets)
        
    # Initialize a dictionary to store predecessors for each class node
    predecessors = {}
    for key_1, value_1 in diz_nodes.items():
        if 'Class' in key_1:
            predecessors[key_1] = []
            for key_2, value_2 in diz_nodes.items():
                if key_1 != key_2 and nx.has_path(dpg_model, value_2, value_1):
                    predecessors[key_1].append(key_2)

    # Calculate the class boundaries
    class_bounds = calculate_boundaries(predecessors)

    # Create a data dictionary to store the extracted metrics
    data = {
        "Communities": asyn_lpa_communities_stack,
        "Class Bounds": class_bounds,
    }

    return data



def get_dpg_node_metrics(dpg_model, nodes_list):
    """
    Extracts metrics from the nodes of a Decision Predicate Graph (DPG) model.

    Args:
    dpg_model: A NetworkX graph representing the Decision Predicate Graph.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    df: A pandas DataFrame containing the metrics for each node in the DPG.
    """
    
    # Calculate the degree of each node
    degree = dict(nx.degree(dpg_model))
    # Calculate the in-degree (number of incoming edges) for each node
    in_nodes = {node: dpg_model.in_degree(node) for node in list(dpg_model.nodes())}
    # Calculate the out-degree (number of outgoing edges) for each node
    out_nodes = {node: dpg_model.out_degree(node) for node in list(dpg_model.nodes())}
    # Calculate the betweenness centrality for each node
    betweness_centrality = nx.betweenness_centrality(dpg_model, weight='weight')
    # Calculate the local reaching centrality for each node
    local_reaching_centrality = {node: nx.local_reaching_centrality(dpg_model, node, weight='weight') for node in list(dpg_model.nodes())}
    
    # Create a dictionary to store the node metrics
    data_node = {
        "Node": list(dpg_model.nodes()),
        "Degree": list(degree.values()),                               # Total degree (in-degree + out-degree)
        "In degree nodes": list(in_nodes.values()),                    # Number of incoming edges
        "Out degree nodes": list(out_nodes.values()),                  # Number of outgoing edges
        "Betweness centrality": list(betweness_centrality.values()),   # Betweenness centrality (useful for identifying bottlenecks)
        "Local reaching centrality": list(local_reaching_centrality.values()),  # Local reaching centrality (useful for feature importance)
    }

    # Merge the node metrics with the node labels
    df = pd.merge(
        pd.DataFrame(data_node),
        pd.DataFrame(nodes_list, columns=["Node", "Label"]),
        on="Node",
        how="left",
    )
    
    # Return the resulting DataFrame
    return df



def get_dpg(X_train, feature_names, model, perc_var, decimal_threshold):
    """
    Generates a Decision Predicate Graph (DPG) from training data and a random forest model.

    Args:
    X_train: A numpy array or similar structure containing the training data samples.
    feature_names: A list of feature names corresponding to the columns in X_train.
    model: A trained random forest model.
    perc_var: A float representing the minimum percentage of total traces a variant must have to be kept.
    decimal_threshold: The number of decimal places to which thresholds are rounded.

    Returns:
    dot: A Graphviz Digraph object representing the Decision Predicate Graph.
    """
    # Initialize an empty log to store the traces
    log = []

    # Trace the paths for each sample in the training data using the random forest model
    for i, sample in enumerate(X_train):
        log.extend(tracing_rf(i, sample, model, feature_names, decimal_threshold))

    # Create a DataFrame from the log
    log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])
    
    # Filter the log based on the variant percentage if specified
    filtered_log = log_df
    if perc_var > 0:
        filtered_log = filter_log(log_df, perc_var)

    # Discover the Data Flow Graph (DFG) from the filtered log
    dfg = discover_dfg(filtered_log)

    # Create a Graphviz Digraph object from the DFG
    dot = generate_dot(dfg, filtered_log)

    # Return the Graphviz Digraph object representing the DPG
    return dot