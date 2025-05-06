import os
import shutil
import re
import ast
import pandas as pd 


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


def parse_class_bounds(raw_text: str) -> dict:
    """
    Parse raw class bounds text into a structured dictionary.

    Args:
        raw_text (str): Raw string of class bounds (Python-like dict format).

    Returns:
        dict: Parsed class bounds with structure {class: {feature: (lower, upper)}}.
    """
    raw_dict = ast.literal_eval(raw_text.split("Class Bounds: ")[-1])
    parsed = {}

    for cls, constraints in raw_dict.items():
        parsed[cls] = {}
        for entry in constraints:
            entry = entry.strip()
            # Match patterns like 'low < feature <= high'
            match = re.match(r"([\d.eE+-]+)\s*<\s*([a-zA-Z0-9_ ]+)\s*<=\s*([\d.eE+-]+)", entry)
            if match:
                low, feat, high = match.groups()
                feat = feat.strip()
                parsed[cls][feat] = (float(low), float(high))
            else:
                # Match patterns like 'feature <= high'
                match = re.match(r"([a-zA-Z0-9_ ]+)\s*<=\s*([\d.eE+-]+)", entry)
                if match:
                    feat, high = match.groups()
                    feat = feat.strip()
                    parsed[cls][feat] = (None, float(high))
                else:
                    # Match patterns like 'low < feature'
                    match = re.match(r"([\d.eE+-]+)\s*<\s*([a-zA-Z0-9_ ]+)", entry)
                    if match:
                        low, feat = match.groups()
                        feat = feat.strip()
                        parsed[cls][feat] = (float(low), None)
                    else:
                        # Match patterns like 'feature > low'
                        match = re.match(r"([a-zA-Z0-9_ ]+)\s*>\s*([\d.eE+-]+)", entry)
                        if match:
                            feat, low = match.groups()
                            feat = feat.strip()
                            parsed[cls][feat] = (float(low), None)

def parse_class_bounds_from_raw(raw_text: str) -> pd.DataFrame:
    """
    Parses a raw string representation of class bounds into a structured DataFrame.

    Parameters:
        raw_text (str): Raw string starting with "Class Bounds: {...}"

    Returns:
        pd.DataFrame: A DataFrame with columns [Class, Feature, Min, Max]
    """
    # Extract the dictionary portion
    class_bounds_str = raw_text.split("Class Bounds: ")[-1]
    class_bounds_dict = ast.literal_eval(class_bounds_str)

    # Pattern for parsing constraints like 'low < feature <= high' or 'feature <= high'
    pattern = re.compile(r"(?:(\d+\.?\d*)\s*<\s*)?([a-zA-Z0-9_ ]+?)\s*(?:<=|<|>)\s*(\d+\.?\d*)")

    records = []
    for cls, constraints in class_bounds_dict.items():
        for constraint in constraints:
            match = pattern.search(constraint)
            if match:
                low, feat, high = match.groups()
                feat = feat.strip()
                records.append({
                    'Class': cls,
                    'Feature': feat,
                    'Min': float(low) if low else None,
                    'Max': float(high)
                })

    return pd.DataFrame(records)

def get_feature_bound_groups_by_class(df_bounds: pd.DataFrame) -> pd.DataFrame:
    """
    Groups classes by shared Min/Max values per feature, to emphasize which groups of classes share values.

    Parameters:
        df_bounds (pd.DataFrame): DataFrame with ['Class', 'Feature', 'Min', 'Max']

    Returns:
        pd.DataFrame: A DataFrame showing grouped class sets for Min and Max per feature
    """
    # Pivot to wide format for comparison
    min_pivot = df_bounds.pivot(index='Feature', columns='Class', values='Min')
    max_pivot = df_bounds.pivot(index='Feature', columns='Class', values='Max')

    def group_classes_by_value(row):
        groups = row.dropna().groupby(row).groups
        return [sorted(list(g)) for g in groups.values()] if len(groups) > 1 else []

    # Apply to both min and max
    min_groups = min_pivot.apply(group_classes_by_value, axis=1)
    max_groups = max_pivot.apply(group_classes_by_value, axis=1)

    # Format into readable strings
    min_groups_fmt = min_groups.apply(lambda lst: "; ".join([", ".join(g) for g in lst]) if lst else "")
    max_groups_fmt = max_groups.apply(lambda lst: "; ".join([", ".join(g) for g in lst]) if lst else "")

    # Combine into final DataFrame
    grouped_df = pd.DataFrame({
        'Min_Classes_Groups': min_groups_fmt,
        'Max_Classes_Groups': max_groups_fmt
    })

    return grouped_df

def parse_communities_from_raw(raw_text: str):
    """
    Parse the raw text representation of a list of sets containing constraints into Python data.

    Args:
        raw_text (str): Raw string containing list of sets with constraint expressions.

    Returns:
        list of sets: Each set contains constraint strings for a community.
    """
    try:
        communities = ast.literal_eval(raw_text.strip())
        if isinstance(communities, list) and all(isinstance(c, set) for c in communities):
            return communities
    except Exception as e:
        print(f"Parsing failed: {e}")
    return []
