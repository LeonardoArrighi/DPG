import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import argparse
import dpg.sklearn_dpg as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="iris", help="Basic dataset to be analyzed")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Random Forest")
    parser.add_argument("--model_name", type=str, default="RandomForestClassifier", help="Chosen tree-based ensemble model")
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG, add the argument to use it as True")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities, add the argument to use it as True")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes, add the argument to use it as True")
    args = parser.parse_args()

    config_path="config.yaml"
    try:
        with open(config_path) as f:
                config = yaml.safe_load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")
    
    pv = config['dpg']['default']['perc_var']
    t = config['dpg']['default']['decimal_threshold']
    j = config['dpg']['default']['n_jobs'] 
    df, df_dpg_metrics = test.test_dpg(datasets = args.ds,
                                        n_learners = args.l, 
                                        perc_var = pv, 
                                        decimal_threshold = t,
                                        n_jobs=j,
                                        model_name = args.model_name,
                                        file_name = os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{pv}_t{t}_stats.txt'), 
                                        plot = args.plot, 
                                        save_plot_dir = args.save_plot_dir, 
                                        attribute = args.attribute, 
                                        communities = args.communities, 
                                        class_flag = args.class_flag)

    df.sort_values(['Degree'])

    df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{pv}_t{t}_node_metrics.csv'),
                encoding='utf-8')

    with open(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{pv}_t{t}_dpg_metrics.txt'), 'w') as f:
        for key, value in df_dpg_metrics.items():
            f.write(f"{key}: {value}\n")
