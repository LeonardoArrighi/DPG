import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math
import os
import numpy as np

import random

from itertools import islice

import graphviz
import networkx as nx

import hashlib
from joblib import Parallel, delayed

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor

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


def tracing_ensemble(case_id, sample, ensemble, feature_names, decimal_threshold=2):
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
    event_log = []

    def build_path(tree, node_index, path):
        node = tree.tree_
        is_leaf = node.children_left[node_index] == node.children_right[node_index]
        feature_index = node.feature[node_index]
        feature_name = feature_names[feature_index]
        threshold = round(node.threshold[node_index], decimal_threshold)
        sample_val = sample[feature_index]

        if is_leaf:
            path.append(f"Class {node.value[node_index].argmax()}")
        else:
            condition = f"{feature_name} <= {threshold}" if sample_val <= threshold else f"{feature_name} > {threshold}"
            path.append(condition)
            next_node = node.children_left[node_index] if sample_val <= threshold else node.children_right[node_index]
            build_path(tree, next_node, path)

    def build_path_reg(tree, node_index, path):
        node = tree.tree_
        if node.children_left[node_index] == node.children_right[node_index]:
            path.append(f"Pred {np.round(node.value[node_index][0], 2)}")
        else:
            feature_index = node.feature[node_index]
            feature_name = feature_names[feature_index]
            threshold = round(node.threshold[node_index], decimal_threshold)
            sample_val = sample[feature_index]
            condition = f"{feature_name} <= {threshold}" if sample_val <= threshold else f"{feature_name} > {threshold}"
            path.append(condition)
            next_node = node.children_left[node_index] if sample_val <= threshold else node.children_right[node_index]
            build_path_reg(tree, next_node, path)

    tree_types = (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
                  AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor)
    if not isinstance(ensemble, tree_types):
        raise Exception("Ensemble model not recognized!")

    for i, tree in enumerate(ensemble.estimators_):
        sample_path = []
        if isinstance(ensemble, (AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor)):
            build_path_reg(tree, 0, sample_path)
        else:
            build_path(tree, 0, sample_path)
        tree_events = [[f"sample{case_id}_dt{i}", step] for step in sample_path]
        event_log.extend(tree_events)

    return event_log

def filter_log(log, perc_var, n_jobs=-1):
    """
    Filters an event log based on the variant percentage. Variants (unique sequences of activities for cases) 
    that occur less than the specified threshold are removed from the log.

    Args:
        log (pd.DataFrame): Event log with columns 'case:concept:name' and 'concept:name'.
        perc_var (float): Minimum percentage of total traces a variant must have to be kept.
        n_jobs (int): Number of parallel jobs (-1 for all CPUs).

    Returns:
        pd.DataFrame: Filtered event log with only valid cases and activities.
    """
    log['case:concept:name'] = log['case:concept:name'].astype(str)
    
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    # Step 1: Compute activity sequences per case (i.e., variants)
    case_variants = log.groupby('case:concept:name')['concept:name'].apply(tuple)
    
    # Step 2: Count unique variant occurrences
    variant_counts = case_variants.value_counts()

    # Step 3: Compute threshold count
    min_occurrences = np.ceil(len(case_variants) * perc_var)

    # Step 4: Identify variants meeting the threshold
    valid_variants = set(variant_counts[variant_counts >= min_occurrences].index)

    # Step 5: Split cases into chunks for parallel processing
    cases_list = [cases for cases in np.array_split(case_variants.index, n_jobs) if len(cases) > 0]

    # Step 6: Function to filter valid cases
    def filter_cases(chunk):
        return [case for case in chunk if case in case_variants.index and case_variants[case] in valid_variants]

    # Step 7: Parallel processing with error handling
    try:
        valid_cases = Parallel(n_jobs=n_jobs)(
            delayed(filter_cases)(chunk) for chunk in cases_list
        )
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        raise

    # Flatten valid cases
    valid_cases = set(sum(valid_cases, []))

    # Step 8: Filter the original log
    filtered_log = log[log['case:concept:name'].isin(valid_cases)].copy()

    return filtered_log

def discover_dfg(log, n_jobs=1):
    """
    Mines the nodes and edges relationships from an event log and returns a dictionary representing
    the Data Flow Graph (DFG). The DFG shows the frequency of transitions between activities.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.
    n_jobs: Number of parallel jobs to use. Default is -1 (use all available CPUs).

    Returns:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.
    """

    # Helper function to process a chunk of cases
    def process_chunk(chunk):
            chunk_dfg = {}
            for case in chunk:
                # Extract the trace (sequence of activities) for the current case
                trace_df = log[log["case:concept:name"] == case].copy()
                trace_df.sort_values(by="case:concept:name", inplace=True)

                # Iterate through the trace to capture transitions between consecutive activities
                for i in range(len(trace_df) - 1):
                    key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])  # Transition
                    if key in chunk_dfg:
                        chunk_dfg[key] += 1  # Increment count if transition exists
                    else:
                        chunk_dfg[key] = 1  # Initialize count if transition is new
            return chunk_dfg

     

    # Get all unique case names
    cases = log["case:concept:name"].unique()
    print("Remaining paths:", len(cases))
    if len(cases) == 0:
       raise Exception("There is no paths with the current value of perc_var and decimal_threshold! Try one less restrictive.") 

    # If n_jobs is -1, use all available CPUs, otherwise use the provided n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count()  # Get the number of available CPU cores
    
    # Ensure n_jobs is at least 1 and no larger than the number of cases
    n_jobs = max(min(n_jobs, len(cases)), 1)  # Ensure n_jobs is within valid range

    # Calculate chunk size, ensure chunk size is at least 1
    chunk_size = max(len(cases) // n_jobs, 1)  # Ensure chunk_size is at least 1
    
    # Split the cases into chunks
    chunks = [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]

    # Process each chunk in parallel
    print("Traversing...")
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)

    # Merge all chunk DFGs into a single DFG dictionary
    print("Aggregating...")
    dfg = {}
    for result in results:
        for key, value in result.items():
            if key in dfg:
                dfg[key] += value  # Aggregate counts for shared transitions
            else:
                dfg[key] = value

    # Return the final DFG dictionary
    return dfg

def generate_dot(dfg):
    """
    Creates a Graphviz directed graph (digraph) from a Data Flow Graph (DFG) dictionary and returns the dot representation.

    Args:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.

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
        if 'Class' or 'Pred' in key:
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
    # Set the random seed for reproducibility
    np.random.seed(42)

    print("Calculating metrics...")
    # Create a dictionary to map node labels to their identifiers
    diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
    # Remove any None keys from the dictionary
    diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
    # Create a reversed dictionary to map node identifiers to their labels
    diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}
    
    # Extract asynchronous label propagation communities
    asyn_lpa_communities = nx.community.asyn_lpa_communities(dpg_model, weight='weight')
    asyn_lpa_communities_stack = [{diz_nodes_reversed[str(node)] for node in community} for community in asyn_lpa_communities]

    filtered_nodes = {k: v for k, v in diz_nodes.items() if 'Class' in k or 'Pred' in k}
    # Initialize the predecessors dictionary
    predecessors = {k: [] for k in filtered_nodes}
    # Find predecessors using more efficient NetworkX capabilities
    for key_1, value_1 in filtered_nodes.items():
        # Using single-source shortest path to find all nodes with paths to value_1
        # This function returns a dictionary of shortest paths to value_1
        try:
            preds = nx.single_source_shortest_path(dpg_model.reverse(), value_1)
            predecessors[key_1] = [k for k, v in diz_nodes.items() if v in preds and k != key_1]
        except nx.NetworkXNoPath:
            continue    

    # Calculate the class boundaries
    print("Calculating constraints...")
    class_bounds = calculate_boundaries(predecessors)

    # Create a data dictionary to store the extracted metrics
    data = {
        "Communities": asyn_lpa_communities_stack,
        "Class Bounds": class_bounds,
    }
    return data



def get_dpg_node_metrics(dpg_model, nodes_list, n_jobs=-1):
    """
    Extracts metrics from the nodes of a DPG model.

    Args:
    dpg_model: A NetworkX graph representing the DPG.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    df: A pandas DataFrame containing the metrics for each node in the DPG.
    """
    print("Calculating node metrics...")
    
    # Initialize dictionaries to store the degrees
    in_nodes = {}
    out_nodes = {}
    degree = {}

    print("Calculating node degree...")
    # Single pass to calculate degrees
    for node in dpg_model.nodes():
        in_nodes[node] = dpg_model.in_degree(node)
        out_nodes[node] = dpg_model.out_degree(node)
        degree[node] = in_nodes[node] + out_nodes[node]

    # Calculate the betweenness centrality for each node
    print("Calculating node betweenness centrality... (100%)")
    sample_size = int(1 * len(dpg_model.nodes()))  # For example, 100% of nodes
    betweenness_centrality = nx.betweenness_centrality(dpg_model, k=sample_size, normalized=True, weight='weight', endpoints=False)

    print(f"Calculating node local reaching centrality... ")
    local_reaching_centrality = {node: nx.local_reaching_centrality(dpg_model, node, weight='weight') for node in list(dpg_model.nodes())}
    
    # Create a dictionary to store the node metrics
    data_node = {
        "Node": list(dpg_model.nodes()),
        "Degree": list(degree.values()),                               # Total degree (in-degree + out-degree)
        "In degree nodes": list(in_nodes.values()),                    # Number of incoming edges
        "Out degree nodes": list(out_nodes.values()),                  # Number of outgoing edges
        "Betweenness centrality": list(betweenness_centrality.values()),   # Betweenness centrality (useful for identifying bottlenecks)
        "Local reaching centrality": list(local_reaching_centrality.values()),  # Local reaching centrality (useful for feature importance)
    }

    # Merge the node metrics with the node labels
    # Assuming data_node and nodes_list are your input data sets
    df_data_node = pd.DataFrame(data_node).set_index('Node')
    df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index('Node')
    df = pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()
    
    # Return the resulting DataFrame
    return df



def get_dpg(X_train, feature_names, model, perc_var, decimal_threshold, n_jobs=-1):
    """
    Generates a DPG from training data and a random forest model.

    Args:
    X_train: A numpy array or similar structure containing the training data samples.
    feature_names: A list of feature names corresponding to the columns in X_train.
    model: A trained random forest model.
    perc_var: A float representing the minimum percentage of total traces a variant must have to be kept. 
    In practical terms, consider a problem with 1,000 paths (learner x training samples), if the minimum percentage threshold is set at 5% (0.05), 
    only those variants that appear in at least 50 of these paths (5% of 1,000) would be retained for further analysis. 
    This helps in reducing the noise and complexity of the data, allowing analysts to concentrate on more common and potentially significant patterns.
    decimal_threshold: The number of decimal places to which thresholds are rounded.
    n_jobs: Number of parallel jobs to run. Default is -1 (use all available CPUs).

    Returns:
    dot: A Graphviz Digraph object representing the DPG.
    """

    print("\nStarting DPG extraction *****************************************")
    print("Model Class:", model.__class__.__name__)
    print("Model Class Module:", model.__class__.__module__)
    print("Model Estimators: ", len(model.estimators_))
    print("Model Params: ", model.get_params())
    print("*****************************************************************")

    def process_sample(i, sample):
        """Process a single sample."""
        return tracing_ensemble(i, sample, model, feature_names, decimal_threshold)

    print('Tracing ensemble...')
    log = Parallel(n_jobs=n_jobs)(
        delayed(process_sample)(i, sample) for i, sample in enumerate(X_train)
    )

    # Flatten the list of lists
    log = [item for sublist in log for item in sublist]
    log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])
    print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')
    
    print(f'Filtering structure... (perc_var={perc_var})')
    # Filter the log based on the variant percentage if specified
    filtered_log = log_df
    if perc_var > 0:
        filtered_log = filter_log(log_df, perc_var)
    
    print('Building DPG...')
    # Discover the Data Flow Graph (DFG) from the filtered log
    dfg = discover_dfg(filtered_log)

    print('Extracting graph...')
    # Create a Graphviz Digraph object from the DFG
    dot = generate_dot(dfg)
    return dot