from graphviz import Digraph, Source
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from PIL import Image 
import re

def paper_plot(plot_name, dot, df):
            
    if df is not None:
        for index, row in df.iterrows():
            #pattern = r'\((?:(?!\().)*\)'
            #row['Label'] = re.sub(pattern, '', row['Label'])
            #print(row['Label'])
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(152, 152, 152))
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(230, 230, 230))
                
    highlight_class_node(dot)

    graph = Source(dot.source)
    graph.render("temp/"+plot_name + "_temp", format="pdf")  # Render as PDF

    with open("temp/" + plot_name + ".dot", "w") as dot_file:
        dot_file.write(dot.source)
        

    
    


def basic_plot(plot_name, dot, df):
            
    if df is not None:

        for index, row in df.iterrows():

            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(122, 122, 122))
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(230, 230, 230))
                
    highlight_class_node(dot)

    graph = Source(dot.source, format="png")
    graph.render("temp/"+plot_name + "_temp", view=False)
    
    dot.render("temp/"+plot_name + "_with_legend", format="pdf")

    # Load the generated image and plot the color bar
    img = Image.open("temp/"+plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    
    plt.savefig("temp/"+plot_name + "_with_legend.png")
    plt.show()


def plot_rf2dpg(plot_name, dot, cn_list):
    colormap = cm.viridis  # Choose a colormap (e.g., viridis)
    norm = None
    if cn_list is not None:
        cn_list = cn_list.sort_values(['CriticalNode'], ascending=True)

        max_score = cn_list['CriticalNodeScore'].max()
        norm = mcolors.Normalize(0, max_score)  # Normalize the scores

        colors = colormap(norm(cn_list['CriticalNodeScore']))  # Assign colors based on scores

        for index, row in cn_list.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['CriticalNode'], color)
    
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
    cbar.set_label('Critical Node Score')

    plt.savefig("temp/"+plot_name + "_with_legend.png")
    plt.show()


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


def change_node_color(graph, node_id, new_color):
    graph.body.append(f'{node_id} [fillcolor="{new_color}"]')

def plot_sample_dpg(plot_name, dot_sample, dot, dtail_list):
    max_score = dtail_list['CriticalNodeScore'].max()
    norm = mcolors.Normalize(0, max_score)  # Normalize the scores

    colormap = cm.viridis  # Choose a colormap (e.g., viridis)
    colors = colormap(norm(dtail_list['CriticalNodeScore']))  # Assign colors based on scores

    for index, row in dtail_list.iterrows():
        color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
        change_node_color(dot, row['CriticalNode'], color)

    if dot_sample is not None:
        for node in dot.body:
            node_name = node.split(' ')[0].replace("\t", "") # Assuming the node name is the first part of the string
            for node_sample in dot_sample.body:
                node_sample_name = node_sample.split(' ')[0].replace("\t", "") # Assuming the node name is the first part of the string
                if node_name==node_sample_name:
                    dot = highlight_node(dot, node_name)
                    #change_node_color(dot, node_sample_name, 'red')


    graph = Source(dot.source, format="png")
    graph.render(plot_name + "_temp", view=False)

    # Load the generated image and plot the color bar
    img = Image.open(plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    cax = plt.axes([0.1, 0.1, 0.8, 0.05])  # Define color bar position
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, orientation='horizontal')
    cbar.set_label('Critical Node Score')

    plt.savefig(plot_name + "_with_legend.png")
    plt.show()

def highlight_class_node(dot):
    for i, line in enumerate(dot.body):
        line_id = line.split(' ')[1].replace("\t", "")
        if line_id.find("Class") != -1:  
            current_color = dot.body[i].split('fillcolor="')[1].split('"')[0]
            dot.body[i] = dot.body[i].replace(current_color, '#a4c2f4').replace("filled", '"rounded, filled" shape=box ')
    return dot

def highlight_node(dot, node_name):
    for i, line in enumerate(dot.body):
        line_id = line.split(' ')[0].replace("\t", "")
        if (node_name == line_id) & (line.find("->") != -1):  
            dot.body[i] = dot.body[i].replace("penwidth=1", 'penwidth=5 color="orange" arrowsize=.7')
            break
    return dot