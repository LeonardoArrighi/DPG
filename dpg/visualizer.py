import os
from graphviz import Source
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from .utils import highlight_class_node, change_node_color, delete_folder_contents

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image
from graphviz import Source

Image.MAX_IMAGE_PIXELS = 500000000  # Adjust based on your needs

def plot_dpg(plot_name, dot, df, df_dpg, save_dir="examples/", attribute=None, communities=False, class_flag=False):
    """
    Plots a Decision Predicate Graph (DPG) with various customization options.

    Args:
    plot_name: The name of the plot.
    dot: A Graphviz Digraph object representing the DPG.
    df: A pandas DataFrame containing node metrics.
    df_dpg: A pandas DataFrame containing DPG metrics.
    save_dir: Directory to save the plot image. Default is "examples/".
    attribute: A specific node attribute to visualize. Default is None.
    communities: Boolean indicating whether to visualize communities. Default is False.
    class_flag: Boolean indicating whether to highlight class nodes. Default is False.

    Returns:
    None
    """
    print("Plotting DPG...")
    # Basic color scheme if no attribute or communities are specified
    if attribute is None and not communities:
        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(157, 195, 230))  # Light blue for class nodes
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(222, 235, 247))  # Light grey for other nodes


    # Color nodes based on a specific attribute
    elif attribute is not None and not communities:
        colormap = cm.Blues  # Choose a colormap
        norm = None

        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing
        
        # Normalize the attribute values if norm_flag is True
        max_score = df[attribute].max()
        norm = mcolors.Normalize(0, max_score)
        colors = colormap(norm(df[attribute]))  # Assign colors based on normalized scores
        
        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
        
        plot_name = plot_name + f"_{attribute}".replace(" ","")
    

    # Color nodes based on community detection
    elif communities and attribute is None:
        colormap = cm.YlOrRd  # Choose a colormap
        
        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing

        # Map labels to community indices
        label_to_community = {label: idx for idx, s in enumerate(df_dpg['Communities']) for label in s}
        df['Community'] = df['Label'].map(label_to_community)
        
        max_score = df['Community'].max()
        norm = mcolors.Normalize(0, max_score)  # Normalize the community indices
        
        colors = colormap(norm(df['Community']))  # Assign colors based on normalized community indices

        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)

        plot_name = plot_name + "_communities"
    

    else:
        raise AttributeError("The plot can show the basic plot, communities or a specific node-metric")

    # Highlight class nodes
    highlight_class_node(dot)

    # Render the graph and display it
    graph = Source(dot.source, format="png")
    graph.render("temp/" + plot_name + "_temp", view=False)

    # Open and display the rendered image
    img = Image.open("temp/" + plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    
    # Add a color bar if an attribute is specified
    if attribute is not None:
        cax = plt.axes([0.11, 0.1, 0.8, 0.025])  # Define color bar position
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, orientation='horizontal')
        cbar.set_label(attribute)

    # Save the plot to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, plot_name + ".png"), dpi=300)
    #plt.show()

    # Clean up temporary files
    delete_folder_contents("temp")

def change_node_color(dot, node, color):
    dot.node(node, style='filled', fillcolor=color)

def normalize_data(df, attribute, colormap):
    norm = Normalize(vmin=df[attribute].min(), vmax=df[attribute].max())
    colors = [colormap(norm(value)) for value in df[attribute]]
    return {node: "#{:02x}{:02x}{:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for node, color in zip(df['Node'], colors)}

def plot_dpg_reg(plot_name, dot, df, df_dpg, save_dir="examples/", attribute=None, communities=False, leaf_flag=False):
    print("Rendering plot...")
    
    node_colors = {}
    if attribute or communities:
        if attribute:
            df = df[~df['Label'].str.contains('Pred')] if leaf_flag else df
            node_colors = normalize_data(df, attribute, plt.cm.Blues)
            plot_name += f"_{attribute.replace(' ', '')}"
        elif communities:
            df['Community'] = df['Label'].map({label: idx for idx, s in enumerate(df_dpg['Communities']) for label in s})
            node_colors = normalize_data(df, 'Community', plt.cm.YlOrRd)
            plot_name += "_communities"
    else:
        base_color = "#9ec3e6" if 'Pred' in df['Label'] else "#dee1f7"
        node_colors = {row['Node']: base_color for index, row in df.iterrows()}

    # Apply node colors
    for node, color in node_colors.items():
        change_node_color(dot, node, color)

    graph_path = os.path.join(save_dir, f"{plot_name}_temp.gv")
    dot.render(graph_path, view=False, format='png')

    # Display and save the image
    img_path = f"{graph_path}.png"
    img = Image.open(img_path)
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.title(plot_name)
    plt.imshow(img)

    if attribute:
        cax = plt.axes([0.11, 0.1, 0.8, 0.025])
        norm = Normalize(vmin=df[attribute].min(), vmax=df[attribute].max())
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=plt.cm.Blues), cax=cax, orientation='horizontal')
        cbar.set_label(attribute)

    plt.savefig(os.path.join(save_dir, f"{plot_name}_REG.png"), dpi=300)
    plt.close()  # Free up memory by closing the plot


    # Clean up temporary files
    delete_folder_contents("temp")