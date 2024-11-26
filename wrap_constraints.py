import os
import argparse
import pandas as pd

import dpg.sklearn_exp as test
from dpg.utils import constraints_generator_class, normalization_f, constraints_viz, create_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="Basic dataset to be analyzed")
    parser.add_argument("--target_column", type=str, help="Name of the column to be used as the target variable")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Random Forest")
    parser.add_argument("--pv", type=float, default=0.001, help="Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.")
    parser.add_argument("--t", type=int, default=2, help="Decimal precision of each feature")
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG, add the argument to use it as True")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities, add the argument to use it as True")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes, add the argument to use it as True")
    args = parser.parse_args()

    
    ds = pd.read_csv(args.ds)

    labels = ds[args.target_column].unique()
    converted_labels = {label: f"Class {i}" for i, label in enumerate(labels)}
    
    constraints_final = {}

    for label in labels:
        model, df, df_dpg_metrics = test.test_base_sklearn(datasets = args.ds,
                                        target_column = args.target_column,
                                        n_learners = args.l, 
                                        perc_var = args.pv, 
                                        decimal_threshold = args.t,
                                        class_constraints = label,
                                        plot = False, 
                                        save_plot_dir = None
                                    )

        constraints_final.update(constraints_generator_class(df_dpg_metrics, converted_labels[label]))

    constraints_dic = create_dic(constraints_final)

    with open(os.path.join(args.dir, f'custom_l{args.l}_pv{args.pv}_t{args.t}_dpg_metrics.txt'), 'w') as f:
        for key, value in constraints_final.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        for key, value in constraints_dic.items():
            f.write(f"{key}: {value}\n")

    if args.plot:
        normalized_constraints, features = normalization_f(constraints_final)
        constraints_viz(constraints_final, args.save_plot_dir, 'output_constraints', {value : key for i, (key, value) in enumerate(converted_labels.items())})

    # df.sort_values(['Degree'])

    # df.to_csv(os.path.join(args.dir, f'custom_l{args.l}_pv{args.pv}_t{args.t}_node_metrics.csv'),
    #             encoding='utf-8')

#python wrap_constraints.py --ds C:\\Users\\leonardo.arrighi\\Documents\\FHG\\fhg\\datasets\\ExperimentMDPI_DPG\\Carambola.csv --target_column Label --l 5 --pv 0.001 --t 2 --dir C:\\Users\\leonardo.arrighi\\Documents\\FHG\\fhg\\constraints\\results
    
        