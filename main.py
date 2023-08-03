import os
import pandas as pd

from data_crunching import crunch_aggregated, save_dataframe

cwd = os.getcwd()


if __name__ == '__main__':
    # size = "P4"
    # category = "CBFM-P"
    sizes = ["P4", "P8", "P12", "P16"]
    categories = ["AC3", "CBFM-P", "RCO"]
    for size in sizes:
        for category in categories:

            df = pd.read_csv(os.path.join(cwd, "dwave_energies", "pegasus", "aggregated", size,
                                          f"{category}_dwave.csv"), sep=";")

            df = crunch_aggregated(df, size, category)

            p = os.path.join(cwd, "data", "pegasus", size, f"{size}_{category}_dwave.csv")
            save_dataframe(df, p)
