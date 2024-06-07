import os
import json
import pickle

import h5py
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from math import isclose
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
instances_base = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
path_to_solutions_base = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies"


def load_instance_to_dataframe(path_to_instance):
    dtype_spec = {
        'i': 'int',
        'j': 'int',
        'v': 'float64'
    }
    instance_df = pd.read_csv(path_to_instance, sep=" ", index_col=False,
                              header=None, comment='#', names=["i", "j", "v"], dtype=dtype_spec)
    return instance_df

def load_instance_to_matrix(path_to_instance):
    dtype_spec = {
        'i': 'int',
        'j': 'int',
        'v': 'float64'
    }
    instance_df = pd.read_csv(path_to_instance, sep=" ", index_col=False,
                              header=None, comment='#', names=["i", "j", "v"], dtype=dtype_spec)
    max_index = max(instance_df[['i', 'j']].max().tolist())
    matrix = np.zeros((max_index, max_index), dtype="float64")
    for row in instance_df.itertuples():
        matrix[row.i-1, row.j-1] = row.v
    zero_rows = np.all(matrix == 0, axis=1)
    holes = np.where(zero_rows)[0]
    return matrix, holes


def create_coo_matrix(matrix: np.ndarray):
    biases = deepcopy(matrix.diagonal())
    np.fill_diagonal(matrix, 0)
    coo = coo_matrix(matrix)
    I = coo.row
    I = np.array([i+1 for i in I]) #change indexing
    J = coo.col
    J = np.array([j+1 for j in J])
    V = np.array(coo.data, dtype="float64")

    return I, J, V, biases


def get_instance_data(path_to_instance):
    matrix, holes = load_instance_to_matrix(path_to_instance)
    I, J, V, biases = create_coo_matrix(matrix)
    return I, J, V, biases, holes


def read_instance_states_energies_from_tn(json_file):
    with open(json_file, encoding='utf-8') as f:
        json_data = json.load(f)
        instance_name = json_data['columns'][json_data['colindex']['lookup']['instance'] - 1][0].split('.')[0]
        instance_name = instance_name.split("_")[0]
        energies = np.array(json_data['columns'][json_data['colindex']['lookup']['drop_eng'] - 1][0])
        states = json_data['columns'][json_data['colindex']['lookup']['ig_states'] - 1][0]

        return instance_name, states, energies,


def merge_states_from_json(data_list, name, holes):
    temp = []
    for dat in data_list:
        if dat[0] == name:
            temp.append((dat[1], dat[2]))
    energies_temp = []
    states_temp = []
    for state_energy in temp:
        for state in state_energy[0]:
            states_temp.append(state)
        for energy in state_energy[1]:
            energies_temp.append(energy)

    energies = np.hstack(energies_temp)
    states_array = []
    for state in states_temp:
        keys = [int(k) for k in state.keys()]
        max_index = max(keys)
        for i in range(1, int(max_index)+1):
            if str(i) not in state.keys():
                state[str(i)] = 0
        state = dict(sorted(state.items(), key=lambda k: int(k[0])))
        temp = [i for i in state.values()]
        states_array.append(temp)
    states = np.vstack(states_array)

    return states, energies


def read_dwave_solutions(path_to_solution):
    df = pd.read_csv(os.path.join(path_to_solution), index_col=0)
    df = df.sort_values(by="energy")
    for field in ["num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"]:
        df = df.drop(field, axis=1)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    energies = df["energy"].to_numpy()
    energies = np.array(energies)
    df = df.drop("energy", axis=1)
    df_dict = df.to_dict(orient='records')

    return energies, df_dict


def create_h5_file(I: np.ndarray, J: np.ndarray, V: np.ndarray, biases: np.ndarray, energies: np.ndarray,
                    states: np.ndarray, path_to_save: str, name: str, metadata: str) -> None:

    with h5py.File(os.path.join(path_to_save, name), "w") as f:
        ising = f.create_group("Ising")
        biases_h5 = ising.create_dataset("biases", data=biases, dtype="float64")
        j_coo = ising.create_group("J_coo")
        i_h5 = j_coo.create_dataset("I", data=I, dtype="i")
        j_h5 = j_coo.create_dataset("J", data=J, dtype="i")
        v_h5 = j_coo.create_dataset("V", data=V, dtype="float64")

        spectrum = f.create_group("Spectrum")
        energies_h5 = spectrum.create_dataset("energies", data=energies, dtype="float64")
        states_h5 = spectrum.create_dataset("states", data=states, dtype="i")

        f.attrs["metadata"] = metadata


def inefective_renumeration(path_to_instance, name):
    df = load_instance_to_dataframe(os.path.join(path_to_instance, name + "_sg.txt"))
    with open(os.path.join(instances_zephyr, name + "_dv.pkl"), "rb") as f:
        h, J = pickle.load(f)
    biases_dict = {}
    for row in df.itertuples():
        if row.i == row.j:
            biases_dict[row.i] = row.v
    renumeration = {}
    for item, value in h.items():
        for key, row_v in biases_dict.items():
            if isclose(value, row_v):
                renumeration[item] = key

def create_h5_from_tn(path_to_instance, path_to_solutions, where_to_save):
    data = []
    for file in os.listdir(path_to_solutions):
        if file.endswith(".json"):
            file_path = os.path.join(path_to_solutions, file)
            name, energies, states = read_instance_states_energies_from_tn(file_path)
            data.append((name, energies, states))

    names = list(set([data_point[0] for data_point in data]))
    for name in names:
        real_name = name + "_sg.txt"
        instance = os.path.join(path_to_instance, real_name)
        I, J, V, biases, holes = get_instance_data(instance)
        states, energies = merge_states_from_json(data, name, holes)


        create_h5_file(I, J, V, biases, energies, states, where_to_save, name + ".hdf5", "SpinGlassPEPS")


def create_h5_from_dw(path_to_instance, path_to_solution, where_to_save, name, size, instance_class):
    path_to_instance = os.path.join(path_to_instance, name + "_sg.txt")
    path_to_solution = os.path.join(path_to_solution, name + ".csv")
    I, J, V, biases, holes = get_instance_data(path_to_instance)
    energies, dict_of_states = read_dwave_solutions(path_to_solution)

    states = []
    for state_dict in dict_of_states:
        state = [i for i in state_dict.values()]
        for hole in reversed(holes):
            state.insert(hole, 0)
        states.append(state)
    states = np.vstack(states)
    create_h5_file(I, J, V, biases, energies, states, where_to_save, f"{size}_{instance_class}_{name}.hdf5", "Advantage_prototype1.1")


def create_h5_from_dw_zephyr(path_to_instance, path_to_solution, where_to_save, name, size, instance_class,
                             renumerate: bool = False):
    path_to_instance_dw = os.path.join(path_to_instance, name + "_dv.pkl")
    path_to_instance = os.path.join(path_to_instance, name + "_sg.txt")
    path_to_solution = os.path.join(path_to_solution, name + ".csv")
    I, J, V, biases, holes = get_instance_data(path_to_instance)
    df = load_instance_to_dataframe(path_to_instance)
    energies, dict_of_states = read_dwave_solutions(path_to_solution)

    states = []
    if not renumerate:
        for state in dict_of_states:
            for hole in holes:
                if str(hole) not in state.keys():
                    state[str(hole)] = 0
        for idx, state_dict in enumerate(dict_of_states):
            state_dict_int = {int(k): v for k, v in state_dict.items()}
            ordered_state_dict = dict(sorted(state_dict_int.items()))
            state = np.array([i for i in ordered_state_dict.values()])
            states.append(state)
    else:
        with open(path_to_instance_dw, "rb") as f:
            h, _ = pickle.load(f)
        nodes = df["i"].unique()
        renumeration = {dw: nodes[idx] for idx, dw in enumerate(h.keys())}
        for idx, state_dict in enumerate(dict_of_states):
            state_dict_renumerated = {renumeration[int(k)]: v for k, v in state_dict.items()}
            state_dict_renumerated_ordered = dict(sorted(state_dict_renumerated.items()))
            state = np.array([i for i in state_dict_renumerated_ordered.values()])
            for hole in holes:
                if hole + 1 not in state_dict_renumerated.keys():
                    state = np.insert(state, hole, 0)
            states.append(state)

    states = np.vstack(states)
    create_h5_file(I, J, V, biases, energies, states, where_to_save, f"{size}_{instance_class}_{name}.hdf5",
                   "Advantage_prototype1.1")


if __name__ == '__main__':
    size = "Z3"
    instances_class = "RAU"
    truncation = "truncate2^12"

    solutions_pegasus_tn = os.path.join(path_to_solutions_base, "pegasus_random_tn", size, instances_class, f"final_bench_{truncation}")
    solutions_zephyr_tn = os.path.join(path_to_solutions_base, "zephyr_random_tn", size, instances_class,
                                        f"final_bench_{truncation}")
    solutions_zephyr_dw = os.path.join(path_to_solutions_base, "aggregated", "zephyr_random", size, instances_class)

    instances_pegasus = os.path.join(instances_base, "pegasus_random", size, instances_class)
    instances_zephyr = os.path.join(instances_base, "zephyr_random", size, instances_class)

    save_pegasus_tn = os.path.join(ROOT, "data", "pegasus", "tn", size, truncation, instances_class)
    save_zephyr_dw = os.path.join(ROOT, "data", "zephyr", "dwave", "DW_" + size, instances_class)
    save_zephyr_tn = os.path.join(ROOT, "data", "zephyr", "tn", size, truncation, instances_class)

    if not os.path.exists(save_zephyr_dw):
        os.makedirs(save_zephyr_dw)

    #create_h5_from_tn(instances_zephyr, solutions_zephyr_tn, save_zephyr_tn)
    for i in tqdm(range(1,101)):
        name = str(i).zfill(3)
        create_h5_from_dw_zephyr(instances_zephyr, solutions_zephyr_dw, save_zephyr_dw, name, size, instances_class, renumerate=True)


