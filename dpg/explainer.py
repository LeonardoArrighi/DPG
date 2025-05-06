import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors

class DecisionPredicateGraphExplainer:
    def __init__(self, dpg):
        print("Initializing DPGExplainer")
        self.dpg_model, self.nodes_list = dpg.to_networkx()
        self.graph_metrics = dpg.extract_graph_metrics(self.dpg_model, self.nodes_list)
        self.df_node_metrics = dpg.extract_node_metrics(self.dpg_model, self.nodes_list)
        self.custom_colors = [
            "#E3C800", "#B09500", "#F0A30A", "#BD7000",
            "#FA6800", "#C73500", "#6D8764", "#3A5431"
        ]

    #def plot_global_explanation(self, save_path=None):
        # Generate plots for class complexity, overlaps, structure
    #    ...

    def parse_class_bounds_strings(self, bounds):
        parsed = {}
        pattern = re.compile(r"(?:(\d+\.?\d*)\s*<\s*)?([a-zA-Z0-9_() ]+)\s*(?:<=|<|>)\s*(\d+\.?\d*)")

        for cls, constraints in bounds.items():
            parsed[cls] = {}
            for c in constraints:
                match = pattern.search(c)
                if match:
                    lo, feat, hi = match.groups()
                    parsed[cls][feat.strip()] = (float(lo) if lo else None, float(hi))
        return parsed



    def plot_class_bounds(self):
        class_bounds = self.graph_metrics["Class Bounds"]
        class_bounds = self.parse_class_bounds_strings(class_bounds)

        # Extract all class labels and all unique features
        classes = sorted(class_bounds.keys())
        all_features = sorted({feat for bounds in class_bounds.values() for feat in bounds})

        # Build row order dynamically: first all *_max, then all *_min
        row_order = [f"{cls}_max" for cls in classes] + [f"{cls}_min" for cls in classes]

        # Build data entries: all max values first, then all min values
        data = []
        for bound_type in ['max', 'min']:  # max classes first
            for cls in classes:
                full_class = f"{cls}_{bound_type}"
                for feat in all_features:
                    lo, hi = class_bounds.get(cls, {}).get(feat, (None, None))
                    val = lo if bound_type == 'min' else hi
                    data.append({'Class': full_class, 'Feature': feat, 'Value': val})

        df = pd.DataFrame(data)
        df_pivot = df.pivot(index='Class', columns='Feature', values='Value')

        # Reorder rows by constructed order
        df_pivot = df_pivot.reindex(row_order)

        # Normalize per feature
        df_norm = df_pivot / df_pivot.max()

        # Create formatted annotations
        annot_data = df_pivot.map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

        # Plot heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            df_norm,
            cmap='YlGn',
            annot=annot_data,
            fmt='',
            cbar_kws={'label': 'Normalized Value'},
            annot_kws={"rotation": 90, "va": "center"}
        )
        plt.title('Normalized Class Bounds per Feature')
        plt.xlabel('Feature')
        plt.ylabel('Class Intervals (max / min)')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save and show
        #output_path = 'class_bounds.pdf'
        #plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_bounds_lines(self):
        """
        Plots class-wise feature bounds using vertical lines (min-max) to emulate a box-plot style.
        Normalizes values per feature to [0, 1], adds caps, spacing, labels, and uses a custom palette.
        """

        # Parse bounds
        class_bounds = self.graph_metrics["Class Bounds"]
        class_bounds = self.parse_class_bounds_strings(class_bounds)
        classes = sorted(class_bounds.keys())
        all_features = sorted({feat for bounds in class_bounds.values() for feat in bounds})
        x = np.arange(len(all_features))

        # Compute normalization ranges
        feature_mins, feature_maxs = {}, {}
        for feat in all_features:
            vals = []
            for cls in classes:
                lo, hi = class_bounds.get(cls, {}).get(feat, (None, None))
                if lo is not None:
                    vals.append(float(lo))
                if hi is not None:
                    vals.append(float(hi))
            feature_mins[feat] = min(vals) if vals else 0
            feature_maxs[feat] = max(vals) if vals else 1

        # Custom palette
        class_colors = {cls: self.custom_colors[i % len(self.custom_colors)] for i, cls in enumerate(classes)}

        # Set up plot
        plt.figure(figsize=(14, 7))
        cap_width = 0.15
        spacing = 0.1
        font_size = 14  # enlarged annotation font

        for idx, cls in enumerate(classes):
            offset = (idx - (len(classes) - 1) / 2) * spacing
            for i, feat in enumerate(all_features):
                lo, hi = class_bounds.get(cls, {}).get(feat, (None, None))
                lo_val = 0 if lo is None else float(lo)
                hi_val = 1 if hi is None else float(hi)
                f_min = feature_mins[feat]
                f_max = feature_maxs[feat]

                if f_max != f_min:
                    lo_norm = (lo_val - f_min) / (f_max - f_min)
                    hi_norm = (hi_val - f_min) / (f_max - f_min)
                else:
                    lo_norm = hi_norm = 0.5

                if not (np.isnan(lo_norm) or np.isnan(hi_norm)):
                    color = class_colors[cls]
                    xi = x[i] + offset

                    # Vertical line
                    plt.plot([xi]*2, [lo_norm, hi_norm], color=color, linewidth=3,
                            label=f'{cls}' if i == 0 else "")

                    # Caps
                    plt.plot([xi - cap_width/2, xi + cap_width/2], [lo_norm, lo_norm], color=color, linewidth=3)
                    plt.plot([xi - cap_width/2, xi + cap_width/2], [hi_norm, hi_norm], color=color, linewidth=3)

                    # Value annotations
                    plt.text(xi + 0.02, lo_norm, f"{lo_val:.2f}", va='bottom', ha='left',
                            fontsize=font_size, color=color)
                    plt.text(xi + 0.02, hi_norm, f"{hi_val:.2f}", va='top', ha='left',
                            fontsize=font_size, color=color)

        plt.title("Class Feature Bounds (Normalized Min-Max with Caps, Spacing and Labels)", fontsize=14)
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Normalized Value", fontsize=12)
        plt.xticks(ticks=x, labels=all_features, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(-0.2, 1.2)
        plt.legend(title="Class", fontsize=10, title_fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def extract_feature(self, constraint: str) -> str:
        match = re.match(r"([a-zA-Z0-9_ ()]+?)\s*[<>]=?\s*[\d.]+", constraint.strip())
        return match.group(1).strip() if match else None

    def plot_communities(self):
        communities = self.graph_metrics["Communities"]
        if not communities:
            print("No communities found in the DPG.")
            return
        # Parse communities
        feature_counts = []
        all_features = set()

        for i, comm in enumerate(communities):
            # Try to find class label
            class_labels = [c for c in comm if c.startswith("Class ")]
            label = class_labels[0] if class_labels else f"Community {i}"

            # Count features in this community
            counter = Counter()
            for rule in comm:
                if rule.startswith("Class "):
                    continue
                feat = self.extract_feature(rule)
                if feat:
                    counter[feat] += 1
                    all_features.add(feat)
            feature_counts.append((label, counter))

        if not all_features:
            print("No features extracted from communities.")
            return

        # Build frequency matrix
        data = []
        for comm_name, counter in feature_counts:
            row = {feat: counter.get(feat, 0) for feat in all_features}
            row['Community'] = comm_name
            data.append(row)

        df_freq = pd.DataFrame(data).set_index('Community')

        if df_freq.empty or df_freq.sum().sum() == 0:
            print("Frequency DataFrame is empty. Cannot plot heatmap.")
            return

        # Plot
        plt.figure(figsize=(1.5 * len(df_freq.columns), 0.6 * len(df_freq)))
        sns.heatmap(df_freq, cmap='YlOrBr', linewidths=0.5, linecolor='gray',
                    cbar=True, annot=True, fmt='d')
        plt.title('Feature Frequency per Class-based Community')
        plt.xlabel('Feature')
        plt.ylabel('Community')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
