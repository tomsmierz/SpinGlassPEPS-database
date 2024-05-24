import h5py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import isclose
from clean_start import load_instance_to_matrix

# PROBLEMS:
# TN P8 CBFM-P
# TN P16 RAU, RCO,


if __name__ == '__main__':

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances_base = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
    dw_pegasus_files = os.path.join(ROOT, "data", "pegasus", "dwave")
    tn_pegasus_files = os.path.join(ROOT, "data", "pegasus", "tn")
    dw_zephyr_files = os.path.join(ROOT, "data", "zephyr", "dwave")
    tn_zephyr_files = os.path.join(ROOT, "data", "zephyr", "tn")

    size = "Z3"
    instances_class = "RAU"
    truncation = "truncate2^16"

    instances_pegasus = os.path.join(instances_base, "pegasus_random", size, instances_class)
    instances_zephyr = os.path.join(instances_base, "zephyr_random", size, instances_class)

    save_pegasus_tn = os.path.join(ROOT, "data", "pegasus", "tn", size, truncation, instances_class)
    save_zephyr_dw = os.path.join(ROOT, "data", "zephyr", "dwave", size, instances_class)

    for file in tqdm(os.listdir(save_zephyr_dw)):
        if file.endswith(".hdf5"):
            file_path = os.path.join(save_zephyr_dw, file)
            with h5py.File(file_path, "r") as f:
                J_coo = f["Ising"]["J_coo"]
                spectrum = f["Spectrum"]

                biases = f["Ising"]["biases"]
                states = spectrum["states"]
                energies = spectrum["energies"]
                I = J_coo["I"]
                J = J_coo["J"]
                V = J_coo["V"]
                max_index = max(max(I), max(J))
                matrix = np.zeros((max_index, max_index), dtype="float64")
                for index2, (i, j) in enumerate(zip(I, J)):
                    matrix[i - 1, j - 1] = V[index2]
                instance_name = file.split(".")[0]
                instance_name = instance_name.split("_")[2]
                matrix2, _ = load_instance_to_matrix(os.path.join(instances_zephyr, instance_name + "_sg.txt"))
                assert np.array_equal(biases, matrix2.diagonal())
                np.fill_diagonal(matrix2, 0)
                assert np.array_equal(matrix, matrix2)
                for index, energy in enumerate(energies):
                    state = states[index, :]
                    linear = np.dot(biases, state)
                    quadratic = np.dot(state, np.dot(matrix2, state.T))
                    assert energy == quadratic + linear
