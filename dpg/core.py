import graphviz
import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math

import numpy as np
np.random.seed(42)

from itertools import combinations
import networkx as nx
from networkx.algorithms import approximation as approx # just for local_node_connectivity 
from collections import Counter

import hashlib
import re

def digraph_to_nx(graphviz_graph):
    networkx_graph = nx.DiGraph()
    nodes_list = []

    edges = []
    weights = {}

    # Extract nodes and edges from the graphviz graph
    for edge in graphviz_graph.body:
        if "->" in edge:
            src, dest = edge.split("->")
            src = src.strip()
            dest = dest.split(" [label=")[0].strip()

            # Extract weight from edge attributes (if available)
            weight = None
            if "[label=" in edge:
                attr = edge.split("[label=")[1].split("]")[0].split(" ")[0]
                weight = (
                    float(attr)
                    if attr.isdigit() or attr.replace(".", "").isdigit()
                    else None
                )
                weights[(src, dest)] = weight

            edges.append((src, dest))

        # Creating the nodes_list
        if "[label=" in edge:
            id, desc = edge.split("[label=")
            id = id.replace("\t", "")
            id = id.replace(" ", "")
            desc = desc.split(" fillcolor=")[0]
            desc = desc.replace('"', "")
            nodes_list.append([id, desc])

    # Sort edges and nodes
    edges = sorted(edges)
    nodes_list = sorted(nodes_list, key=lambda x: x[0])

    # Add nodes and edges to the NetworkX graph
    for edge in edges:
        src, dest = edge
        if (src, dest) in weights:
            networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
        else:
            networkx_graph.add_edge(src, dest)

    return networkx_graph, nodes_list


def tracing_rf(case_id, sample, rf_classifier, feature_names, decimal_threshold=1):
    event_log = []
    def build_path(tree, node_index, path=[]):
        tree_ = tree.tree_
        if tree_.children_left[node_index] == tree_.children_right[node_index]:
            path.append(f"Class {tree_.value[node_index].argmax()}")
        else:
            feature_name = feature_names[tree_.feature[node_index]]
            threshold = round(float(tree_.threshold[node_index]), decimal_threshold)
            sample_val = sample[tree_.feature[node_index]]

            if sample_val <= threshold:
                path.append(f"{feature_name} <= {round(float(tree_.threshold[node_index]), decimal_threshold)}")
                build_path(tree, tree_.children_left[node_index], path)
            else:
                path.append(f"{feature_name} > {round(float(tree_.threshold[node_index]), decimal_threshold)}")
                build_path(tree, tree_.children_right[node_index], path)

    for i, tree_in_forest in enumerate(rf_classifier.estimators_):
        sample_path = []
        build_path(tree_in_forest, 0, sample_path)
        for step in sample_path:
            event_log.append(["sample" + str(case_id) + "_dt" + str(i), step])

    return event_log


def filter_log(log, perc_var):
    """
    Filters log based on variant percentage (variants that occurred less than threshold are removed)
    """
    variants = {}
    for case in log["case:concept:name"].unique():
        key = "|".join(
            [x for x in log[log["case:concept:name"] == case]["concept:name"]]
        )
        if key in variants:
            variants[key].append(case)
        else:
            variants[key] = [case]
    total_traces = log["case:concept:name"].nunique()

    cases, activities = [], []
    for k, v in variants.items():
        if len(v) / total_traces >= perc_var:
            for case in v:
                for act in k.split("|"):
                    cases.append(case)
                    activities.append(act)
        else:
            continue
    log = pd.DataFrame(
        zip(cases, activities), columns=["case:concept:name", "concept:name"]
    )
    return log


def discover_dfg(log):
    """
    Mines the nodes and edges relationships and returns a dictionary
    """
    dfg = {}
    for case in log["case:concept:name"].unique():
        trace_df = log[log["case:concept:name"] == case].copy()
        trace_df.sort_values(by="case:concept:name", inplace=True)
        for i in range(len(trace_df) - 1):
            key = (trace_df.iloc[i, 1], trace_df.iloc[i + 1, 1])
            if key in dfg:
                dfg[key] += 1
            else:
                dfg[key] = 1
    return dfg


def generate_dot(dfg, log):
    """
    Creates a graphviz digraph from graph (dictionary) and returns the dot
    """
    act_count = Counter(log["concept:name"])
    dot = graphviz.Digraph(
        "FHG",
        engine="dot",
        graph_attr={
            "bgcolor": "white",
            "rankdir": "R",
            "overlap": "false",
            "fontsize": "20",
        },
        node_attr={"shape": "box"},
    )
    added_nodes = set()
    
    sorted_dict_values = {k: v for k, v in sorted(dfg.items(), key=lambda item: item[1])}
    # Sort the items of the dfg dictionary for deterministic order
    for k, v in sorted_dict_values.items():
        
        if k[0] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
                #label=f"{k[0]} ({act_count[k[0]]})", #do not put the count (for paper)
                label=f"{k[0]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[0])
        if k[1] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
                #label=f"{k[1]} ({act_count[k[1]]})",  #do not put the count (for paper)
                label=f"{k[1]}",                
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[1])
        dot.edge(
            str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
            label=str(v), penwidth="1", fontsize="18"
        )
    return dot


def get_fhg_metrics(fhg_model, nodes_list):
    """
    Extract metrics from FHG
    """
    

    diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
    diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
    diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}
    
    # -------------- NON FUNZIONA --------------
    local_node_connectivity = []
    for key_1, value_1 in diz_nodes.items():
        for key_2, value_2 in diz_nodes.items():
            if ('Class' in key_1) and ('Class' in key_2) and (key_1 != key_2):
                local_node_connectivity.append([key_1, key_2, approx.local_node_connectivity(fhg_model, value_1, value_2)])

    bridges = nx.bridges(fhg_model.to_undirected())
    bridges_stack = []
    for sets in bridges:
        new_sets = set()
        for node in sets:
            new_sets.add(diz_nodes_reversed[str(node)])
        bridges_stack.append(new_sets)
    
    cut_vertex = nx.articulation_points(fhg_model.to_undirected())
    cut_vertex_stack = []
    for vertex in cut_vertex:
        if 'Class' not in str(diz_nodes_reversed[str(vertex)]):
            cut_vertex_stack.append(diz_nodes_reversed[str(vertex)])
    
    
    weakly_connected_components = nx.number_weakly_connected_components(fhg_model)

    asyn_lpa_communities = nx.community.asyn_lpa_communities(fhg_model, weight='weight') # # find communities
    asyn_lpa_communities_stack = []
    for sets in asyn_lpa_communities:
        new_sets = set()
        for node in sets:
            new_sets.add(diz_nodes_reversed[str(node)])
        asyn_lpa_communities_stack.append(new_sets)

    # convert communities in a dictionary
    asyn_lpa_communities_dict = {}
    for sets in asyn_lpa_communities_stack:
        key_turn = None
        for key in sets:
            if 'Class' in key:
                asyn_lpa_communities_dict[key] = []
                key_turn = key
                
        for element in sets:
            if 'Class' not in element:
                if key_turn is not None:
                    if key_turn in asyn_lpa_communities_dict:
                        asyn_lpa_communities_dict[key_turn].append(element)

    print(len(asyn_lpa_communities_dict['Class 1']))    
    asyn_lpa_communities_bounds = calculate_boundaries(asyn_lpa_communities_dict)
    print(len(asyn_lpa_communities_bounds['Class 1']))


    overall_reciprocity = nx.overall_reciprocity(fhg_model)

    if nx.is_directed_acyclic_graph(fhg_model):

        ancestors = {}
        for key, value in diz_nodes.items():
            if ('Class' in key):
                sets = nx.ancestors(fhg_model, value)
                new_sets = set()
                for node in sets:
                    new_sets.add(diz_nodes_reversed[str(node)])
                ancestors[key] = new_sets

        descendants = {}
        for key, value in diz_nodes.items():
            if (fhg_model.in_degree(value) == 0):
                sets = nx.descendants(fhg_model, value)
                new_sets = set()
                for node in sets:
                    new_sets.add(diz_nodes_reversed[str(node)])
                descendants[key] = new_sets

        common = []
        for key_1, value_1 in diz_nodes.items():
            for key_2, value_2 in diz_nodes.items():
                if ('Class' in key_1) and ('Class' in key_2) and (key_1 != key_2):
                    common.append([key_1, key_2, diz_nodes_reversed[str(nx.lowest_common_ancestor(fhg_model, value_1, value_2))]])
        
        # symple_cycles = 0
    
    else:
        ancestors = descendants = common = "The DHG model is not directed acyclic."

        #symple_cycles = len([i for i in nx.simple_cycles(fhg_model)])
        
    predecessors = {}
    for key_1, value_1 in diz_nodes.items():
        if ('Class' in key_1):
            predecessors[key_1] = []
            for key_2, value_2 in diz_nodes.items():
                if (key_1 != key_2) and nx.has_path(fhg_model, value_2, value_1):
                    predecessors[key_1].append(key_2)

    class_bounds = calculate_boundaries(predecessors)





    data = {
        "Local Node Connectivity": local_node_connectivity,
        "Bridges": bridges_stack,
        "Cut-Vertex": cut_vertex_stack,
        "Weakly Connected Components": weakly_connected_components,
        "Communities": asyn_lpa_communities_stack,
        "Communities Bounds": asyn_lpa_communities_bounds,
        "Ancestors": ancestors,
        "Descendants": descendants,
        "Lowest Common Nodes (Classes)": common,
        #"Cycles": symple_cycles,
        "Overall Reciprocity": overall_reciprocity,
        "Predecessors": predecessors,
        "Class Bounds": class_bounds,
    }

    return data

def calculate_boundaries(dict):
    # script for decide boundaries of every feature of every class
    boundaries_class = {}
    for key, value in dict.items():
        if ('Class' in key):

            boundaries_class[key] = []
            
            key_set = []
            for i in dict[key]:
                key_set.append(str(re.split(' <= | > ', i)[0]))
            key_set = set(key_set)
            
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

                # save as string the boundaries
                alfa = None
                if match_list[0] == math.inf:
                    alfa = str(valore_unico + " <= " + str(match_list[1]))
                elif match_list[1] == -math.inf:
                    alfa = str(valore_unico + " > " + str(match_list[0]))
                else:
                    alfa = str(str(match_list[0]) + ' < ' + valore_unico + ' <= ' + str(match_list[1]))
                boundaries_class[key].append(alfa)
    return boundaries_class


def get_fhg_node_metrics(fhg_model, nodes_list):
    """
    Extract metrics from FHG's nodes
    """

    degree = dict(nx.degree(fhg_model))
    closeness = nx.closeness_centrality(fhg_model)
    d_nodes = {node : fhg_model.degree(node) for node in list(fhg_model.nodes())}
    d_centrality = nx.degree_centrality(fhg_model)
    in_nodes = {node : fhg_model.in_degree(node) for node in list(fhg_model.nodes())}
    in_centrality = nx.in_degree_centrality(fhg_model)
    out_nodes = {node : fhg_model.out_degree(node) for node in list(fhg_model.nodes())}
    out_centrality = nx.out_degree_centrality(fhg_model)
    bottleneck_centrality = {node : in_centrality[node] / out_centrality[node] if out_centrality[node] != 0 else 0 for node in in_centrality}
    eigenvector_centrality = nx.eigenvector_centrality(fhg_model, max_iter=10000, weight = 'weight')
    betweness_centrality = nx.betweenness_centrality(fhg_model, weight='weight')
    #katz_centrality = nx.katz_centrality(fhg_model, max_iter = 1000, normalized = True, weight = 'weight')
    local_reaching_centrality = {node : nx.local_reaching_centrality(fhg_model, node, weight = 'weight') for node in list(fhg_model.nodes())}
    constraint = nx.constraint(fhg_model, weight='weight')
    
    
    # Create a DataFrame with node metrics
    data_node = {
        "Node": list(fhg_model.nodes()), # # !
        "Degree": list(degree.values()), # # !
        "Closeness": list(closeness.values()), # # !
        "Degree nodes": list(d_nodes.values()),                                 # # edges linked
        "Degree centrality": list(d_centrality.values()),                       # # edges linked / # all edges 
        "In degree nodes": list(in_nodes.values()),                             # # in edges linked
        "In degree centrality": list(in_centrality.values()),                   # # in edges linked / # all edges
        "Out degree nodes": list(out_nodes.values()),                           # # out edges linked
        "Out degree centrality": list(out_centrality.values()),                 # # out edges linked / # all edges
        "Bottleneck centrality": list(bottleneck_centrality.values()),          # # in / out
        "Eigenvector centrality": list(eigenvector_centrality.values()),        # # useful for bottlenecks
        "Betweness centrality": list(betweness_centrality.values()),            # # useful for bottlenecks
        # "Katz centrality": list(katz_centrality.values()),                      # # useful for path importance
        "Local reaching centrality": list(local_reaching_centrality.values()),  # # idea for pruning
        "Constraint" : list(constraint.values()),                               # # A high constraint value for a node implies that the node is a key connector within a group of nodes that have strong ties to each other. In other words, if v is highly constrained, it means that its neighbors are not only connected to v but also form a tightly knit cluster themselves.
        
    }


    df = pd.merge(
        pd.DataFrame(data_node),
        pd.DataFrame(nodes_list, columns=["Node", "Label"]),
        on="Node",
        how="left",
    )
    return df


def get_fhg(X_train, feature_names, model, perc_var, decimal_threshold):
    log = []
    for i, sample in enumerate(X_train):
        log.extend(tracing_rf(i, sample, model, feature_names, decimal_threshold))

    log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])
    
    filtered_log = log_df
    if perc_var>0:
        filtered_log = filter_log(log_df, perc_var)

    dfg = discover_dfg(filtered_log)

    # Create a Graphviz Digraph object from the DFG graph
    dot = generate_dot(dfg, filtered_log)

    return dot


def get_label(id, nodes_list):
    #print('id',id)
    return [node[1] for node in nodes_list if node[0] == id][0]

def get_target_classes(str_targets):
    df = pd.DataFrame(str_targets.split('|'), columns=['Targets'])
    df[['Target1_Label', 'Target2_Label']] = df['Targets'].str.split('_x_', expand=True)
    df.drop(columns=['Targets'], inplace=True)
    df = df.iloc[1:]
    return df

def find_last_common_node(path1, path2):
    for node1 in reversed(path1[1:]):
        for node2 in reversed(path2[1:]):
            if node1 == node2:
                return(node1)
    return None

def get_critical_nodes(df, fhg_model, nodes_list, n_estimators, n_training_samples, verbose=False):
    df = df.sort_values(['Degree']).reset_index().iloc[:,1:]
    prefix = "Class"
    matching_items = df[df["Label"].str.startswith(prefix)]["Node"]
    possible_roots = df[df["Closeness"] == 0].Node.values

    combs = list(combinations(matching_items, 2))
    cn_list = []

    nodes_list = sorted(nodes_list, key=lambda x: x[0])
    #print(nodes_list)
    
    combs.sort()  # Sorting a list directly

    # Searching for critical nodes
    for root in possible_roots:
        if not get_label(root, nodes_list).startswith(prefix):
            source_node = root
            for node in combs:
                target_node_1 = node[0]
                target_node_2 = node[1]
                
                # Calculate the shortest paths between the specified nodes
                try:
                    shortest_path_1 = nx.shortest_path(
                        fhg_model, source=source_node, target=target_node_1
                    )
                except nx.NetworkXNoPath as e:
                    # Handle the exception here
                    if verbose:
                        print(
                            "No path between "
                            + get_label(source_node, nodes_list)
                            + " (source) and "
                            + get_label(target_node_1, nodes_list)
                            + " (target)."
                        )
                    continue

                try:
                    shortest_path_2 = nx.shortest_path(
                        fhg_model, source=source_node, target=target_node_2)
                    
                except nx.NetworkXNoPath as e:
                    # Handle the exception here
                    if verbose:
                        print(
                            "No path between "
                            + get_label(source_node, nodes_list)
                            + " (source) and "
                            + get_label(target_node_2, nodes_list)
                            + " (target)."
                        )
                    continue

                # Find the common node (last split node) between the two paths
                last_common_node = find_last_common_node(shortest_path_1, shortest_path_2)
                if last_common_node == None:
                    continue
                
                dist_last_common_node_1 = nx.shortest_path_length(
                    fhg_model, source=last_common_node, target=target_node_1
                )
                dist_last_common_node_2 = nx.shortest_path_length(
                    fhg_model, source=last_common_node, target=target_node_2
                )

                # Find the weight to the closest nodes (out-branch)
                dist_last_common_node_1_w = nx.shortest_path_length(
                    fhg_model,
                    source=last_common_node,
                    target=target_node_1,
                    weight="weight",
                )
                dist_last_common_node_2_w = nx.shortest_path_length(
                    fhg_model,
                    source=last_common_node,
                    target=target_node_2,
                    weight="weight",
                )

                node_score = compute_critical_node_score(dist_last_common_node_1,
                    dist_last_common_node_2,
                    dist_last_common_node_1_w,
                    dist_last_common_node_2_w,
                    n_estimators,
                    n_training_samples
                )

                
                if verbose:
                    print(
                        "The last common node (DTAIL) between "
                        + get_label(target_node_1, nodes_list)
                        + " and "
                        + get_label(target_node_2, nodes_list)
                        + " the two paths is: "
                        + get_label(last_common_node, nodes_list)
                    )

                cn_list.append(
                    [
                        target_node_1,
                        get_label(target_node_1, nodes_list),
                        target_node_2,
                        get_label(target_node_2, nodes_list),
                        last_common_node,
                        get_label(last_common_node, nodes_list),
                        dist_last_common_node_1_w + dist_last_common_node_2_w,
                        dist_last_common_node_1 + dist_last_common_node_2,
                        node_score,
                    ]
                )

    

    # No CN was found
    if len(cn_list)==0:
        return None, None

    cn_list.sort(key=lambda x: (x[4], x[5]))  # Sort based on CriticalNode and its label
    
    cn_list = pd.DataFrame(cn_list)
    cn_list.columns = [
        "Node1",
        "Node1Label",
        "Node2",
        "Node2Label",
        "CriticalNode",
        "CriticalNodeLabel",
        "SumDist",
        "SumWeight",
        "CriticalNodeScore",
    ]
    cn_list.sort_values(["CriticalNodeScore"], ascending=False).to_csv(
        "CriticalNodeScore_all_edges.csv",
        index=False
    )
    
    if verbose:
        print('cn_list',cn_list)

    cn_list_summary = (
        cn_list.groupby(["CriticalNode", "CriticalNodeLabel"])["CriticalNodeScore"]
        .mean()
        .reset_index()
        .sort_values(["CriticalNodeScore"], ascending=False)
    )
    
    targets = []
    for idx, row in cn_list_summary.iterrows():
        targets_value = ""
        for cn_idx, cn_row in cn_list[cn_list['CriticalNode'] == row['CriticalNode']].iterrows():
            targets_value += "|" + cn_row["Node1Label"] + '_x_' + cn_row["Node2Label"]
        targets.append(targets_value)

    cn_list_summary["Targets"] = targets
    #df_dtail_summary.to_csv("CriticalNodeScore.csv", index=False)
    if verbose:
        print('cn_list_summary', cn_list_summary)

    
    return cn_list_summary, cn_list


def compute_critical_node_score(dist_1, dist_2, weight_1, weight_2, n_estimators, n_training_samples):
    # Calculate weights for individual paths
    weight_ratio_1 = weight_1 / (n_estimators * n_training_samples)
    weight_ratio_2 = weight_2 / (n_estimators * n_training_samples)

    # Calculate scores for individual paths
    score_1 = weight_ratio_1 / dist_1
    score_2 = weight_ratio_2 / dist_2

    # Compute the total critical node score
    total_score = score_1 + score_2

    return total_score


def shortest_path_with_node(graph, source, target, intermediate_node):
    try:
        path_source_to_intermediate = nx.shortest_path(graph, source, intermediate_node)                    
    except nx.NetworkXNoPath as e:
        return None
    try:
        path_intermediate_to_target = nx.shortest_path(graph, intermediate_node, target)                  
    except nx.NetworkXNoPath as e:
        return None
    
    # Combine the paths, ensuring the intermediate node is used
    if intermediate_node in path_source_to_intermediate and intermediate_node in path_intermediate_to_target:
        path_source_to_intermediate.remove(intermediate_node)  # Remove the duplicate intermediate node
        shortest_path = path_source_to_intermediate + path_intermediate_to_target
        return shortest_path
    return None

def critical_nodes_performance(df,fhg_model,cn_list, nodes_list, X_train):
    if cn_list is None:
        return None

    df_cn_perf = X_train
    #print('features', df_cn_perf.columns)
    df_cn_perf.columns = [str(col).rstrip().replace(' ', '_') for col in df_cn_perf.columns]
    df_cn_perf.columns =  [re.sub(r'_\([^)]*\)', '', string) for string in df_cn_perf.columns]
    df_cn_perf.columns =  [re.sub(r'/', '', string) for string in df_cn_perf.columns]


    #print('features', df_cn_perf.columns)


    df = df.sort_values(['Degree']).reset_index().iloc[:,1:]
    prefix = "Class"
    possible_roots = df[df["Closeness"] == 0].Node.values

    nodes_list = sorted(nodes_list, key=lambda x: x[0])

    # Checking Critical Nodes Coverage
    cn_list_perf = []
    for i, cn in cn_list.iterrows():
        #print("****")
        #print(cn["CriticalNodeLabel"])
        #print('CNS', cn["CriticalNodeScore"])
        #print('cn ', cn)
        for root in possible_roots:
            if not get_label(root, nodes_list).startswith(prefix):
                source_node = root
                required_node = cn["CriticalNode"]
                target_node = cn["Node1"]
                path = shortest_path_with_node(fhg_model, source_node, target_node, required_node)
                if path != None:
                    total_samples, true_predictions, current_class = get_path_classification(df, path, df_cn_perf, cn["Node1Label"])
                    cn_list_perf.append([source_node, required_node, cn["CriticalNodeLabel"], np.round(cn["CriticalNodeScore"],4), target_node, total_samples, true_predictions, current_class, cn["Node1Label"]])

                target_node = cn["Node2"]
                path = shortest_path_with_node(fhg_model, source_node, target_node, required_node)
                if path != None:
                    total_samples, true_predictions, current_class = get_path_classification(df, path, df_cn_perf, cn["Node2Label"])
                    cn_list_perf.append([source_node, required_node, cn["CriticalNodeLabel"], np.round(cn["CriticalNodeScore"],4), target_node, total_samples, true_predictions, current_class, cn["Node2Label"]])
    cn_list_perf = pd.DataFrame(cn_list_perf, columns=["RootNode", "CriticalNode", "CriticalNodeLabel", "CriticalNodeScore","TargetNode", "TotalSamples", "TruePrediction", "Class", "ClassNode"])
    cn_list_perf = cn_list_perf.sort_values(["CriticalNodeScore"], ascending=False)
    #cn_list_perf.to_csv("cn_list_perf.csv")
    #print(cn_list_perf)
    return cn_list_perf

    



            

def get_path_classification(df, path, df_cn_perf, target_node):
    current_class = int(re.sub(r'\([^)]*\)', '', target_node).replace("Class ",""))

    df_cn_paths = df[df.Node.isin(path)]
    expressions = [re.sub(r'\([^)]*\)', '', string.rstrip()) for string in df_cn_paths["Label"].values.astype(str)]

    pattern =  r'(\b\w+)\s+(\S+)\s+(\S+)'
    expressions_with_underscore = [re.sub(pattern, r'\1_\2_\3', value) for value in expressions]
    #print('expressions_with_underscore', expressions_with_underscore)

    # Create a function to classify based on the logical expressions
    def classify(row):
        results = []
        for expr in expressions_with_underscore:
            if expr.startswith("Class"):
                expr = expr.rstrip().replace(" ", "==").replace("Class", "target")
            expr = re.sub(r'(_)([><=]+)', r' \2', expr)
            expr = re.sub(r'([><=]+)(_)', r' \1', expr)
            expr = expr.replace("/", "").replace("Class", "target")

            result = eval(expr, row.to_dict())  # Pass the row data as a dictionary
            results.append(result)
        return all(results)
    
    df_cn_perf['Predicted_Class'] = df_cn_perf[df_cn_perf["target"]==current_class].apply(classify, axis=1)
    
    total_samples = df_cn_perf[df_cn_perf['target']==current_class].shape[0]
    true_predictions = df_cn_perf[df_cn_perf['Predicted_Class']==True].shape[0]
    #print('Total',total_samples)
    #print('Predicted',true_predictions)
    return total_samples, true_predictions, current_class
