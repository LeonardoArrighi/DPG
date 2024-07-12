import os
import argparse

import dpg.sklearn_standard_dpg as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="iris", help="Basic dataset to be analyzed")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the RF")
    parser.add_argument("--pv", type=float, default=0.001, help="Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.")
    parser.add_argument("--t", type=int, default=1, help="Decimal precision of each feature")
    parser.add_argument("--dir", type=str, default="test/", help="Folder to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the FHG, add the argument to use it as True")
    # parser.add_argument("--png", action='store_true', help="Save the FHG, add the argument to use it as True")
    args = parser.parse_args()


    df, df_fhg_metrics, length = test.test_base_sklearn(datasets = args.ds , 
                                            n_learners = args.l,
                                            perc_var = args.pv,
                                            decimal_threshold = args.t,
                                            file_name = os.path.join(args.dir, f'iris_l{args.l}_pv{args.pv}_t{args.t}_stats.txt'),
                                            # dtail_name = os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_dtail.txt'),
                                            # plot = True,
                                            plot = args.plot,
                                            # save_png = args.png,
                                            # save_folder = args.dir
                                            )

# nda: length la teniamo per quando re-inseriremo i critical nodes
    
    
    df.sort_values(['Degree'])

    df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_node_metrics_(crit{length}).csv'),
                encoding='utf-8')
    # print(df[['Local reaching centrality', 'Label']].sort_values(by=['Local reaching centrality']))
    with open(os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_fhg_metrics_(crit{length}).txt'), 'w') as f:
        for key, value in df_fhg_metrics.items():
            riga = f"{key}: {value}\n"
            f.write(riga)
        
    
# python -W ignore to_work_with.py --ds iris --l 5 --pv 0 --t 2 --plot --dir C:\\Users\\leonardo.arrighi\\Documents\\FHG\\R\\results\\metrics\\test