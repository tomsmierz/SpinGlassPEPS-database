import os
import pandas as pd
import re
from tqdm import tqdm

cwd = os.getcwd()


def crunch_aggregated(data_frame: pd.DataFrame, size: str, category: str) -> pd.DataFrame:
    """
    Transforms (crunch) aggregated dataframe from old format into new format
    :param data_frame: aggregated pandas.Dataframe in old format
    :param size: topology and size of instances (ex. P8)
    :param category: Category of instances (ex. AC3)
    :return: Transformed dataframe in new format
    """
    df_crunched = data_frame.drop(["Unnamed: 0", "instance"], axis=1)
    index = [f"{size}_{category}_{i}" for i in range(1, 101)]
    df_crunched["instance"] = index
    df_crunched.rename(columns={"best_dwave": "energy", "probability_dwave [%]": "probability [%]"}, inplace=True)
    col = df_crunched.columns.tolist()
    col = col[-1:] + col[:-1]
    df_crunched = df_crunched[col]
    df_crunched.set_index("instance", inplace=True)
    return df_crunched


def save_dataframe(data_frame: pd.DataFrame, path: str) -> None:
    data_frame.to_csv(path)


def aggregate(path: str) -> pd.DataFrame:

    best_dwave = []
    prob_best = []
    inst_name = []

    for i in tqdm(range(1, 101)):
        num = f"00{i}"[-3::]
        c = 0
        min_values = []
        probs = []
        for file in os.listdir(path):
            if num in file:
                df = pd.read_csv(os.path.join(path, file))
                df_min = df[df.energy == df.energy.min()]
                prob = df_min.num_occurrences.sum() / df.num_occurrences.sum()
                prob = round(100*prob, 2)

                min_values.append(df_min.energy[0])
                probs.append(prob)

        inst_name.append(num)
        best = min(min_values)
        best_index = min_values.index(best)
        prob_of_min = probs[best_index]
        best_dwave.append(best)
        prob_best.append(prob_of_min)


    data = {"instance": inst_name, "best_dwave": best_dwave, "probability_dwave [%]": prob_best}
    aggregated_results = pd.DataFrame(data)
    save_path = os.path.join(cwd, f"..\\energies\\aggregated\\{size}")
    name = f"{folder}_dwave.csv"
    aggregated_results.to_csv(os.path.join(save_path, name), sep=";")
    print(aggregated_results)


