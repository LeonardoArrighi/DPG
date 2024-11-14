import os
import argparse

import dpg.sklearn_standard_dpg as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="iris", help="Basic dataset to be analyzed")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Random Forest")
    parser.add_argument("--pv", type=float, default=0.001, help="Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.")
    parser.add_argument("--t", type=int, default=2, help="Decimal precision of each feature")
    parser.add_argument("--model_name", type=str, default="RandomForestClassifier", help="Chosen tree-based ensemble model")
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG, add the argument to use it as True")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities, add the argument to use it as True")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes, add the argument to use it as True")
    args = parser.parse_args()



    df, df_dpg_metrics = test.test_base_sklearn(datasets = args.ds,
                                        n_learners = args.l, 
                                        perc_var = args.pv, 
                                        decimal_threshold = args.t, 
                                        model_name = args.model_name,
                                        file_name = os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_stats.txt'), 
                                        plot = args.plot, 
                                        save_plot_dir = args.save_plot_dir, 
                                        attribute = args.attribute, 
                                        communities = args.communities, 
                                        class_flag = args.class_flag)

    df.sort_values(['Degree'])

    df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_node_metrics.csv'),
                encoding='utf-8')

    with open(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_dpg_metrics.txt'), 'w') as f:
        for key, value in df_dpg_metrics.items():
            f.write(f"{key}: {value}\n")