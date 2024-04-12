import numpy as np
np.random.seed(42)
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from .core import get_target_classes

import matplotlib.pyplot as plt
import plotly.graph_objects as go




def importance_vs_critical(rf_classifier, cn_list, data, feature_names):
    cn_list.to_csv("cn_list.csv")
    importances = rf_classifier.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    top_n = 1
    important_feature = feature_names[sorted_indices[:top_n][0]]
    
    for f in feature_names:
        if cn_list["CriticalNodeLabel"][0].startswith(f):
            critical_feature = f
    
    critical_interval = get_interval_from_node_label(cn_list["CriticalNodeLabel"][0].replace(critical_feature, ""))

    sns.scatterplot(data=data, x=critical_feature, y=important_feature, hue=data.columns[-1])
    plt.axvline(x=critical_interval, color='r', linestyle='--', label="Critical Value") 
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()  # Show the plot




def get_interval_from_node_label(string):
    replacements = {
        '>': '',  # Example: Replace '∧' with '&'
        '<': '',  # Example: Replace '∨' with '|'
        '=': '',  # Example: Replace '¬' with '!'
    }
    # Create a regular expression pattern to match the symbols
    pattern = re.compile('|'.join(re.escape(symbol) for symbol in replacements.keys()))
    replaced_string = pattern.sub(lambda match: replacements[match.group(0)], string)

    pattern = re.compile(r"[-+]?\d*\.\d+|\d+")  # Matches float numbers, e.g., 1.23, -4.567, 123
    matches = pattern.findall(replaced_string)
    if matches:
        return float(matches[0])  # Convert the matched string to a float
    else:
        return None  # R


def enriched_rf_importance(rf_classifier, cn_list, feature_names):
    feature_importances = rf_classifier.feature_importances_

    feature_importances = pd.DataFrame([feature_importances, feature_names]).T
    feature_importances.columns = ["Feature Importance", "Feature Name"]
    
    cn_importances = cn_list
    cn_importances[['Feature Name', 'Node Criteria', 'Number of Paths']] = cn_importances['CriticalNodeLabel'].str.extract(r'(.+?)\s([<>]=?\s\d+\.\d+)\s\((\d+)\)')

    feature_importances = pd.merge(feature_importances, cn_importances, on='Feature Name', how='left')

    df = pd.DataFrame(feature_importances)
    df = df.sort_values(by='Feature Importance', ascending=False)

    # Drop rows with NaN in CriticalNodeScore
    df['CriticalNodeScore'].fillna(0, inplace=True)
    df['Node Criteria'].fillna(0, inplace=True)
    df['CriticalNodeScore'] = df['CriticalNodeScore'].round(2)

    df_sorted = df.sort_values(by='Feature Importance', ascending=False)

    # Plot using Seaborn relplot
    sns.set(style='whitegrid')
    g = sns.relplot(x='CriticalNodeScore', y='Feature Name', hue='CriticalNodeScore',
                    sizes=(00, 500), alpha=1, palette="viridis", 
                    size="CriticalNodeScore", data=df_sorted, height=6, aspect=1.5, legend=None)
    g.set_axis_labels('Critical Node Score / Feature Importance', 'Feature Name')
    plt.title('Feature Importance and Critical Node Score')

    #for index, row in df_sorted.iterrows():
    #    if not pd.isna(row["Targets"]):
    #        aux = 0
    #        for i, current_row in get_target_classes(row["Targets"]).iterrows():
    #            t1 = current_row["Target1_Label"]
    #            t2 = current_row["Target2_Label"]
    #            plt.text(row['CriticalNodeScore'], index+aux, t1+' and '+t2, color='black', va='center')
    #            aux = aux + 0.12
            
   
    norm = Normalize(vmin=0, vmax=df['CriticalNodeScore'].max())
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # Fake empty array for the ScalarMappable
    cbar = plt.colorbar(sm)
    cbar.set_label('Scale of Critical Node Score')

    plt.barh(df_sorted['Feature Name'], df_sorted['Feature Importance'], color='gray', alpha=0.3, label='Feature Importance')
    for i, v in enumerate(df_sorted['Feature Importance'].unique()):
        plt.text(v, i, f'{v:.2f}', color='black', va='center')

    plt.legend(frameon=False)
    plt.show()

def importance_vs_criticalscore(rf_classifier, cn_list, feature_names):
    feature_importances = rf_classifier.feature_importances_

    feature_importances = pd.DataFrame([feature_importances, feature_names]).T
    feature_importances.columns = ["Feature Importance", "Feature Name"]
    
    cn_importances = cn_list
    cn_importances[['Feature Name', 'Node Criteria', 'Number of Paths']] = cn_importances['CriticalNodeLabel'].str.extract(r'(.+?)\s([<>]=?\s\d+\.\d+)\s\((\d+)\)')

    feature_importances = pd.merge(feature_importances, cn_importances, on='Feature Name', how='left')

    df = pd.DataFrame(feature_importances)
    df = df.sort_values(by='Feature Importance', ascending=False)
    df.dropna(inplace=True)
    df['CriticalNodeScore'] = df['CriticalNodeScore'].round(2)
    df['Number of Paths'] = df['Number of Paths'].astype(int)

    # Plot using Seaborn relplot
    sns.set(style='whitegrid')
    sns.scatterplot(x='CriticalNodeScore', y='Feature Importance', hue='CriticalNodeScore',
                    palette="viridis",data=df, size='Number of Paths', sizes=(10, 100))
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.title('')

    for index, row in df.iterrows():
        plt.text(row['CriticalNodeScore'], row['Feature Importance'],
                  str(row['Feature Name'])+'\n'+str(row['Node Criteria']),
                  ha='center', va='bottom', fontsize=8,
                  rotation=45)

    # Add a legend
    legend = plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.get_frame().set_linewidth(0) 
    plt.tight_layout()  # Ensures the plot and legend fit within the figure area properly
    plt.show()

    
def criticalscores_class(cn_list):
    #cn_list = cn_list.sort_values(["CriticalNodeScore"], ascending=False).reset_index()
    source = []
    target = []
    value = []
    cns_values = []
    classes = set()

    for i, r in cn_list.iterrows():
        cns = np.round(r["CriticalNodeScore"], 2)
        cn_label = r["CriticalNodeLabel"]
        cn_targets = get_target_classes(r["Targets"])

        for _, cn_row in cn_targets.iterrows():
            source.append(cn_row[0])
            target.append(cn_label)
            value.append(cns)
            cns_values.append(cns)

            source.append(cn_label)
            target.append(cn_row[1])
            value.append(cns)
            cns_values.append(cns)

            if cn_row[0].startswith("Class"):  # Check if the node label starts with "Class"
                classes.add(cn_label)
            if cn_row[1].startswith("Class"):  # Check if the node label starts with "Class"
                classes.add(cn_label)

    # Create a DataFrame with source, target, and value
    df = pd.DataFrame({"source": source, "target": target, "values": value})

    # Grouping by source and target to sum values for duplicate links
    df = df.groupby(['source', 'target']).mean().reset_index()

    # Creating nodes and links
    nodes = pd.Series(pd.concat([df['source'], df['target']]).unique())
    links = df.apply(lambda row: nodes[nodes == row['source']].index[0], axis=1), \
            df.apply(lambda row: nodes[nodes == row['target']].index[0], axis=1), \
            df['values']
    
    # Assign colors to nodes based on condition
    colors = []
    for node in nodes:
        if node in classes:  # Nodes starting with "Class" in blue
            colors.append("#FFFFFF")
        else:  # Others in red
            colors.append("#a4c2f4")

    #print('link', links)

    colormap = cm.get_cmap('viridis')
    norm = plt.Normalize(0, max(links[2]))
    c = [colormap(norm(value)) for value in links[2]]
    hex_color = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in c]

    # Plotting the Sankey diagram with node colors
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=colors  # Color based on condition
        ),
        link=dict(
            source=links[0],
            target=links[1],
            value=links[2],
            color=hex_color,  # Color based on intensity
            hovertemplate='%{source.label} -> %{target.label} : %{value}<extra></extra>'  # Hover template
        )
    )])

    # Title and layout settings
    fig.update_layout(title_text="Sankey Diagram with Class and Critical Nodes")
    fig.show()
