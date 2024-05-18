import os
import h5py
import numpy as np
import pandas as pd

from typing import Optional
from data_creation.renumeration import advantage_6_1_to_spinglass_int
from data_creation.utils import h5_tree


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


def create_h5_file_from_dwave(path_to_data: str, path_to_instance: str, path_to_save: str, instance_name: str,
                              data_name: str, file_name: str, size: int) -> None:
    dtype_spec = {
        'i': 'int',
        'j': 'int',
        'v': 'float64'
    }
    pd.set_option('display.precision', 20)
    instance_df = pd.read_csv(os.path.join(path_to_instance, instance_name), sep=" ", index_col=False,
                              header=None, comment='#', names=["i", "j", "v"], dtype=dtype_spec)
    biases = []
    I = []
    J = []
    V = []
    for row in instance_df.itertuples():
        if row.i == row.j:
            biases.append(row.v)
        else:
            I.append(row.i)
            J.append(row.j)
            V.append(row.v)
    biases = np.array(biases)
    I = np.array(I)
    J = np.array(J)
    V = np.array(V)

    df = pd.read_csv(os.path.join(path_to_data, data_name), index_col=0)
    df = df.sort_values(by="energy")
    for field in ["num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"]:
        df = df.drop(field, axis=1)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    energies = df["energy"].to_numpy()
    energies = np.array(energies)
    df = df.drop("energy", axis=1)
    df_dict = df.to_dict(orient='records')
    columns = []
    for row in df_dict:
        temp = {advantage_6_1_to_spinglass_int(int(i), size): k for i, k in row.items()}
        temp = dict(sorted(temp.items()))
        columns.append([i for i in temp.values()])
    states = np.column_stack(columns)
    _create_h5_file(I, J, V, biases, energies, states, path_to_save, file_name, "Advantage_system6.1")



if __name__ == '__main__':
    path_dwave = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies\aggregated\pegasus_random\P4\AC3"
    instance_path = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances\pegasus_random\P4\AC3"
    save_path_h5 = r"C:\Users\walle\PycharmProjects\SpinGlassPEPS-database\data\pegasus\dwave"
    create_h5_file_from_dwave(path_dwave, instance_path, save_path_h5, "001_sg.txt", "001.csv", "P4_AC3_001.hdf5", 4)



