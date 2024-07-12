import os
import shutil



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
