from graphviz import Source
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from PIL import Image
from dpg.utils import highlight_node, highlight_class_node, change_node_color, delete_folder_contents



def basic_plot(plot_name, dot, df, save_dir=None, attribute=None, norm_flag=False, class_flag=True):
    
    if attribute == None:    
    
        for index, row in df.iterrows():

            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(157, 195, 230))
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(222, 235, 247))
                    
        highlight_class_node(dot)

        graph = Source(dot.source, format="png")
        
        graph.render("temp/"+plot_name + "_temp", view=False)
        
        # Load the generated image and plot the color bar
        img = Image.open("temp/"+plot_name + "_temp.png")
        plt.figure(figsize=(16, 8))
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title(plot_name)
        plt.imshow(img)
        
        plt.savefig("examples/"+plot_name + ".png")
        plt.show()

        delete_folder_contents("temp")





def plot_custom_map(plot_name, dot, df, attribute, norm_flag=False, class_flag=True):
    
    colormap = cm.bwr  # Choose a colormap (e.g., viridis)
    
    norm = None
    
    if df is not None:
        # df = df.sort_values(['Node'], ascending=True)
        
        if not class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#f7ef79')
            df = df[~df.Label.str.contains('Class')]
            df = df.reset_index(drop=True)
            
        if norm_flag:
            max_score = df[attribute].max()
            
            norm = mcolors.Normalize(0, max_score)  # Normalize the scores
            
            colors = colormap(norm(df[attribute]))  # Assign colors based on scores
            
        else:
            colors = colormap(df[attribute])  # Assign colors based on scores

        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
        
    highlight_class_node(dot)

    graph = Source(dot.source, format="png")
    graph.render("temp/"+plot_name + "_temp", view=False)

    # Load the generated image and plot the color bar
    img = Image.open("temp/"+plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    cax = plt.axes([0.1, 0.1, 0.8, 0.05])  # Define color bar position
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, orientation='horizontal')
    cbar.set_label(attribute)

    plt.savefig("temp/"+plot_name + "_with_legend.png")
    plt.show()


def plot_communities_map(plot_name, dot, df, communities_list):
    
    colormap = cm.rainbow  # Choose a colormap (e.g., viridis)
        
    if df is not None:

        label_to_community = {label: idx for idx, s in enumerate(communities_list) for label in s}
        df['Community'] = df['Label'].map(label_to_community)
        
        max_score = df['Community'].max()
            
        norm = mcolors.Normalize(0, max_score)  # Normalize the scores
        
        colors = colormap(norm(df['Community']))  # Assign colors based on scores

        for index, row in df.iterrows():

            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
                
    highlight_class_node(dot)

    graph = Source(dot.source, format="png")
    graph.render("temp/"+plot_name + "_temp", view=False)

    # Load the generated image and plot the color bar
    img = Image.open("temp/"+plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    
    plt.savefig("temp/"+plot_name + "_with_legend.png")
    plt.show()