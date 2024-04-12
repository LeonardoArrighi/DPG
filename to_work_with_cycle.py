import os
import argparse
from tqdm import tqdm

import fhg.test_sklearn_work as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="iris", help="Basic dataset to be analyzed")
    # parser.add_argument("--l", type=int, default=5, help="Number of learners for the RF")
    # parser.add_argument("--pv", type=float, default=0.001, help="Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.")
    # parser.add_argument("--t", type=int, default=1, help="Decimal precision of each feature")
    parser.add_argument("--dir", type=str, default="test/", help="Folder to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the FHG, add the argument to use it as True")
    # parser.add_argument("--png", action='store_true', help="Save the FHG, add the argument to use it as True")
    args = parser.parse_args()

    for l in tqdm([1, 2, 5, 10, 20, 50, 75, 100]):
        for pv in [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
            for t in [0]:#, 1, 2]:

                try:
                    df, length = test.test_base_sklearn(datasets = args.ds , 
                                                n_learners = l,
                                                perc_var = pv,
                                                decimal_threshold = t,
                                                file_name = os.path.join(args.dir, f'{args.ds}_l{l}_pv{pv}_t{t}_stats.txt'),
                                                # dtail_name = os.path.join(args.dir, f'{args.ds}_l{args.l}_pv{args.pv}_t{args.t}_dtail.txt'),
                                                plot = args.plot,
                                                # save_png = args.png,
                                                # save_folder = args.dir
                                                )
                    df.sort_values(['Degree'])

                    df.to_csv(os.path.join(args.dir, f'{args.ds}_l{l}_pv{pv}_t{t}_metrics_(crit{length}).csv'),
                            encoding='utf-8')
                except:
                    pass    
# python to_work_with_cycle.py --ds iris --dir C:\\Users\\leonardo.arrighi\\Documents\\FHG\\R\\results\\metrics\\iris