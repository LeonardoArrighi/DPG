import os
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np



def highlight_class_node(dot):
    """
    Highlights nodes in the Graphviz Digraph that contain "Class" in their identifiers by changing their fill color
    and adding a rounded shape.

    Args:
    dot: A Graphviz Digraph object.

    Returns:
    dot: The modified Graphviz Digraph object with the class nodes highlighted.
    """
    # Iterate over each line in the dot body
    for i, line in enumerate(dot.body):
        # Extract the node identifier from the line
        line_id = line.split(' ')[1].replace("\t", "")
        # Check if the node identifier contains "Class"
        if "Class" in line_id:
            # Extract the current fill color of the node
            current_color = dot.body[i].split('fillcolor="')[1].split('"')[0]
            # Replace the current color with the new color and add rounded shape attribute
            dot.body[i] = dot.body[i].replace(current_color, '#a4c2f4').replace("filled", '"rounded, filled" shape=box ')
    
    # Return the modified Graphviz Digraph object
    return dot



def change_node_color(graph, node_id, new_color):
    """
    Changes the fill color of a specified node in the Graphviz Digraph.

    Args:
    graph: A Graphviz Digraph object.
    node_id: The identifier of the node whose color is to be changed.
    new_color: The new color to be applied to the node.

    Returns:
    None
    """
    # Append a new line to the graph body to change the fill color of the specified node
    graph.body.append(f'{node_id} [fillcolor="{new_color}"]')



def delete_folder_contents(folder_path):
    """
    Deletes all contents of the specified folder.

    Args:
    folder_path: The path to the folder whose contents are to be deleted.

    Returns:
    None
    """
    # Iterate over each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)  # Get the full path of the item
        try:
            # Check if the item is a file or a symbolic link
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove the file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove the directory and its contents
        except Exception as e:
            # Print an error message if the deletion fails
            print(f'Failed to delete {item_path}. Reason: {e}')


#########


def max_store(feat):
    max_num = max(x for x in feat if x is not None)
    return max_num

def min_store(feat):
    min_num = min(x for x in feat if x is not None)
    return min_num


def constraints_generator_class(df, class_label):
    constraints = {}
    constraints[class_label] = []
    for constraint in df['Class Bounds'][class_label]:
        constraint = re.split(r'(<=|>|<|>=)', constraint)
        if len(constraint) == 3:
            if constraint[1] == '>':
                constraints[class_label].append({'feature' : constraint[0].replace(' ',''), 'min' : float(constraint[2]), 'max' : None})            
            elif constraint[1] == '<=':
                constraints[class_label].append({'feature' : constraint[0].replace(' ',''), 'min' : None, 'max' : float(constraint[2])})
        elif len(constraint) == 5:    
            constraints[class_label].append({'feature' : constraint[2].replace(' ',''), 'min' : float(constraint[0]), 'max' : float(constraint[4])})
    constraints = dict(sorted(constraints.items()))
    return {classe: sorted(valori, key=lambda x: x['feature']) for classe, valori in constraints.items()}


def features_values(constraints):
    features = set()
    for key in constraints:
        for constraint in constraints[key]:
            features.add(constraint['feature'])
    
    store_m = {}
    for feature in features:   
        store_m[feature] = []
        for key in constraints:
            for constraint in constraints[key]:
                if (constraint['feature'] == feature):
                    store_m[feature].append(constraint['min'])
                    store_m[feature].append(constraint['max'])
    return features, store_m


def normalization_f(constraints):

    features, store_m = features_values(constraints)
    
    for key in constraints:
        for constraint in constraints[key]:
            for feature in features:
                if constraint['feature'] == feature:
                    if constraint['min'] == None:
                        constraint['min'] = 0.0
                        try:
                            constraint['max'] = ((constraint['max'] - min_store(store_m[feature])) / (max_store(store_m[feature]) - min_store(store_m[feature])))
                        except:
                            constraint['max'] = 1.0
                    elif constraint['max'] == None:
                        try:
                            constraint['min'] = ((constraint['min'] - min_store(store_m[feature])) / (max_store(store_m[feature]) - min_store(store_m[feature])))
                        except:
                            constraint['min'] = 0.0
                        constraint['max'] = 1.0
                    elif constraint['min'] != None and constraint['max'] != None:
                        constraint['min'] = ((constraint['min'] - min_store(store_m[feature])) / (max_store(store_m[feature]) - min_store(store_m[feature])))
                        constraint['max'] = ((constraint['max'] - min_store(store_m[feature])) / (max_store(store_m[feature]) - min_store(store_m[feature])))
    
    return constraints, sorted(features)



def constraints_viz(constraints, name, save_dir, class_names):  

    normalized_constraints, features = normalization_f(constraints)
    # Custom colors and class names

    colors = {'Class 0': '#6169ae', 'Class 1': '#f17a8f', 'Class 2': '#ffbb6b', 'Class 3': '#444760'}

    # Get all unique features across classes
    all_features = sorted(set(feature['feature'] for cls_data in normalized_constraints.values() for feature in cls_data))

    # Prepare data for plotting
    x_labels = all_features
    feature_min_max = {cls: {entry['feature']: (entry['min'], entry['max']) for entry in entries} for cls, entries in normalized_constraints.items()}

    # Plot with increased spacing
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.2
    x_indices = np.arange(len(x_labels)) * 1.5  # Increase spacing between features

    for i, (original_cls, new_cls) in enumerate(class_names.items()):
        min_vals, max_vals = [], []
        for feature in x_labels:
            if feature in feature_min_max[original_cls]:
                min_val, max_val = feature_min_max[original_cls][feature]
                min_vals.append(min_val)
                max_vals.append(max_val)
            else:
                min_vals.append(0)
                max_vals.append(0)
        
        # Plot range bars with custom colors
        bar_positions = x_indices + i * bar_width
        bars = ax.bar(bar_positions, [max_val - min_val for min_val, max_val in zip(min_vals, max_vals)], 
                    bar_width, bottom=min_vals, label=new_cls, alpha=0.7, color=colors[original_cls])
        
        # Highlight single-point bars and annotate missing features
        for bar, min_val, max_val, feature in zip(bars, min_vals, max_vals, x_labels):
            if min_val == max_val:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
            if min_val == max_val == 0:
                bar.set_color('black')
                bar.set_alpha(0.8)
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + min_val + 0.02, 
                        'Feature is missing', rotation=90, ha='center', va='bottom', 
                        fontsize=8, color='black')

    # Add a legend entry for single-point ranges
    single_point_legend = plt.Line2D([0], [0], color='white', marker='s', markerfacecolor='white', 
                                    markeredgewidth=2, markeredgecolor='red', linestyle='', label='Single point')

    # Formatting plot
    ax.set_xticks(x_indices + bar_width * (len(class_names) - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Value Range')
    ax.set_title(f'Normalized constraints')
    ax.legend(handles=[plt.Line2D([0], [0], marker='s', color=colors[original_cls], markersize=10, label=new_cls) 
                    for original_cls, new_cls in class_names.items()] + [single_point_legend])
    plt.grid(linestyle=':')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, name + ".png"))


def create_dic(constraints):
    constraints_dic = {}
    for key in constraints:
        constraints_dic[key] = []
        for item in constraints[key]:
            if item['max'] == None:
                constraints_dic[key].append(str(item['min'])+' <= '+item['feature'])
            elif item['min'] == None:
                constraints_dic[key].append(item['feature']+ ' < '+str(item['max']))
            else:
                constraints_dic[key].append(str(item['min'])+' <= '+item['feature']+ ' < '+str(item['max']))
    return constraints_dic