import os
import h5py
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Optional
from scipy.sparse import coo_matrix
from data_creation.renumeration import advantage_6_1_to_spinglass_int
from data_creation.utils import h5_tree, array_from_dict


def aggregate_all(dwave_source: str, size: int, sbm_source: str, tn_source: str) -> None:
    result = pd.DataFrame(columns=["instance", "best_found_state", "best_found_energy", "DW_energy",
                                   "SBM_energy", "TN_energy", "source"])
    best_state_dw, best_energy_dw = load_dwave(dwave_source, size)


def load_dwave(path_to_instance: str, size: int) -> (dict, float):
    df = pd.read_csv(os.path.join(path_to_instance), index_col=0)
    df = df.sort_values(by="energy")
    best = df.iloc[0]
    best_energy = best.energy
    best_state = {}
    for column, data in best.iteritems():
        if column.isdigit():
            best_state[advantage_6_1_to_spinglass_int(int(column), size)] = data
    best_state = dict(sorted(best_state.items()))
    return best_state, best_energy


def _create_h5_file(I: np.ndarray, J: np.ndarray, V: np.ndarray, biases: np.ndarray, energies: np.ndarray,
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


def _read_instance(path_to_instance: str, instance_name: str) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype_spec = {
        'i': 'int',
        'j': 'int',
        'v': 'float64'
    }
    pd.set_option('display.precision', 20)
    instance_df = pd.read_csv(os.path.join(path_to_instance, instance_name), sep=" ", index_col=False,
                              header=None, comment='#', names=["i", "j", "v"], dtype=dtype_spec)
    max_index = max(instance_df[['i', 'j']].max().tolist())
    dense_matrix = np.zeros((max_index, max_index), dtype="float64")
    biases_dic = {}
    biases = []
    I = []
    J = []
    V = []
    for row in instance_df.itertuples():
        if row.i == row.j:
            biases_dic[row.i] = row.v
        else:
            dense_matrix[row.i-1, row.j-1] = row.v

    for index in range(1, max_index+1):
        if index in biases_dic.keys():
            biases.append(biases_dic[index])
        else:
            biases.append(0)

    for i in range(dense_matrix.shape[0]):
        for j in range(dense_matrix.shape[1]):
            element = dense_matrix[i, j]
            if element != 0:
                I.append(i+1)
                J.append(j+1)
                V.append(element)

    biases = np.array(biases)
    I = np.array(I)
    J = np.array(J)
    V = np.array(V)

    return I, J, V, biases


def create_h5_file_from_dwave(path_to_data: str, path_to_instance: str, path_to_save: str, instance_name: str,
                              data_name: str, file_name: str, size: int, zephyr=False) -> None:

    I, J, V, biases = _read_instance(path_to_instance, instance_name)
    df = pd.read_csv(os.path.join(path_to_data, data_name), index_col=0)
    df = df.sort_values(by="energy")
    for field in ["num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"]:
        df = df.drop(field, axis=1)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    energies = df["energy"].to_numpy()
    energies = np.array(energies)
    df = df.drop("energy", axis=1)
    max_index = max(max(I), max(J))
    if zephyr:
        pass

    else:
        renumerated = {i: advantage_6_1_to_spinglass_int(int(i), size) for i in df.columns}
        df.rename(columns=renumerated, inplace=True)
        df = df[sorted(df.columns)]
    df_dict = df.to_dict(orient='records')
    rows = []

    for row in df_dict:
        if not zephyr:
            state = []
            for index in range(1, max_index+1):
                if index in row.keys():
                    state.append(row[index])
                else:
                    state.append(0)
        else:
            indices = sorted(list(set(list(I) + list(J))))
            state = []
            row_values = iter(row.values())
            for index in range(1, max_index + 1):
                if index in indices:
                    spin = next(row_values)
                    state.append(spin)
                else:
                    state.append(0)

        rows.append(state)

    states = np.vstack(rows)
    _create_h5_file(I, J, V, biases, energies, states, path_to_save, file_name, "Advantage_system6.1")


def create_h5_from_tn(path_to_data: str, path_to_instance: str, path_to_save: str):

    for filename in tqdm(os.listdir(path_to_data)):
        file = os.path.join(path_to_data, filename)
        if os.path.isfile(file) and file.endswith(".json"):
            with open(file, encoding='utf-8') as f:
                json_data = json.load(f)
                instance_name = json_data['columns'][json_data['colindex']['lookup']['instance'] - 1][0].split('.')[0]
                instance_name = instance_name.split("_")[0]
                energies = np.array(json_data['columns'][json_data['colindex']['lookup']['drop_eng'] - 1][0])
                states = json_data['columns'][json_data['colindex']['lookup']['ig_states'] - 1][0]
                states = array_from_dict(states)
                saved_file_path = os.path.join(path_to_save, instance_name + ".hdf5")
                if os.path.isfile(os.path.join(saved_file_path, instance_name + '.hdf5')):
                    with h5py.File(saved_file_path, "r") as f:

                        spectrum = f["Spectrum"]
                        states_read = np.array(spectrum["states"])
                        energies_read = np.array(spectrum["energies"])
                        indices_of_added_rows = []
                        added_energies = []
                        for i, row in enumerate(states):
                            is_in_file = np.any(np.all(states_read == row, axis=1))

                            if not is_in_file:
                                indices_of_added_rows.append(i)
                                added_energies.append(energies[i])
                                energies_read = np.append(energies_read, energies[i])
                                states_read = np.vstack([states_read, row])
                        indices_to_sort = np.argsort(energies_read)
                        energies = energies_read[indices_to_sort]
                        states = states_read[indices_to_sort]
                    I, J, V, biases = _read_instance(path_to_instance, instance_name + "_sg.txt")

                else:
                    I, J, V, biases = _read_instance(path_to_instance, instance_name + "_sg.txt")
                _create_h5_file(I, J, V, biases, energies, states, path_to_save, instance_name + ".hdf5",
                                    "SpinGlassPEPS.jl")


if __name__ == '__main__':
    instances_base = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dw_pegasus_files = os.path.join(ROOT, "data", "pegasus", "dwave")
    tn_pegasus_files = os.path.join(ROOT, "data", "pegasus", "tn")
    dw_zephyr_files = os.path.join(ROOT, "data", "zephyr", "dwave")
    tn_zephyr_files = os.path.join(ROOT, "data", "zephyr", "tn")

    instance_class = "CBFM-P"
    truncation = "truncate2^16"
    dw_p16_data = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies\aggregated\pegasus_random\P16\{instance_class}"
    tn_p8_data = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies\pegasus_random_tn\P8\{instance_class}\final_bench_{truncation}"
    size = "P8"
    dw_z3_data = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies\aggregated\zephyr_random\Z3\{instance_class}"

    instance_name = os.path.join(instances_base, "pegasus_random", size, instance_class)
    save = os.path.join(dw_zephyr_files, size, instance_class)
    save_tn = os.path.join(tn_pegasus_files, size, truncation, instance_class)
    # for file in tqdm(os.listdir(dw_z3_data)):
    #     if file.endswith(".csv"):
    #         name = file.split(".")[0]
    #         create_h5_file_from_dwave(dw_z3_data, instance_name, save, name + "_sg.txt", file, f"{size}_{instance_class}_{name}.hdf5", 3, zephyr=True)

    create_h5_from_tn(tn_p8_data, instance_name, save_tn)