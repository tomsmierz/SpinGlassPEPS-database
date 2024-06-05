import os
import pickle

import numpy as np
import networkx as nx
import matplotlib as mpl

from clean_start import read_instance_states_energies_from_tn, load_instance_to_matrix, load_instance_to_dataframe, \
    merge_states_from_json, read_dwave_solutions
from copy import deepcopy
from math import isclose
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
instances_base = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
dw_pegasus_files = os.path.join(ROOT, "data", "pegasus", "dwave")
tn_pegasus_files = os.path.join(ROOT, "data", "pegasus", "tn")
dw_zephyr_files = os.path.join(ROOT, "data", "zephyr", "dwave")
tn_zephyr_files = os.path.join(ROOT, "data", "zephyr", "tn")

path_to_solutions_base = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies"
size = "Z3"
instances_class = "RAU"
truncation = "truncate2^16"

solutions_pegasus_tn = os.path.join(path_to_solutions_base, "pegasus_random_tn", size, instances_class, f"final_bench_{truncation}")
instances_pegasus = os.path.join(instances_base, "pegasus_random", size, instances_class)

solutions_zephyr_dw = os.path.join(path_to_solutions_base, "aggregated", "zephyr_random", size, instances_class)


instances_zephyr = os.path.join(instances_base, "zephyr_random", size, instances_class)

save_pegasus_tn = os.path.join(ROOT, "data", "pegasus", "tn", size, truncation, instances_class)
save_zephyr_dw = os.path.join(ROOT, "data", "zephyr", "dwave", size, instances_class)


TN = False
if __name__ == '__main__':
    if TN:
        for file in os.listdir(solutions_pegasus_tn):
            if file.endswith(".json"):
                file_path = os.path.join(solutions_pegasus_tn, file)
                name, states_dict, energies = read_instance_states_energies_from_tn(file_path)
                data = [(name, states_dict, energies)]
                df = load_instance_to_dataframe(os.path.join(instances_pegasus, name + "_sg.txt"))
                matrix, holes = load_instance_to_matrix(os.path.join(instances_pegasus, name + "_sg.txt"))
                biases = deepcopy(matrix.diagonal())
                np.fill_diagonal(matrix, 0)
                states, energies = merge_states_from_json(data, name, holes)

                for index, state_dict in enumerate(states_dict):
                    state = states[index, :]
                    assert len(state) == len(biases)
                    for key, value in state_dict.items():
                        spin_in_state = state[int(key)-1]
                        assert spin_in_state == value
                    en = 0
                    for row in df.itertuples():
                        if row.i == row.j:
                            en += state_dict[str(row.i)] * row.v
                        else:
                            en += state_dict[str(row.i)] * state_dict[str(row.j)] * row.v
                    assert isclose(en, energies[index])

                for index, energy in enumerate(energies):
                    state = states[index, :]
                    linear = np.dot(state, biases)
                    quadratic = np.dot(state.T, np.dot(matrix, state))
                    assert energy == linear + quadratic
    else:
        for file in tqdm(os.listdir(solutions_zephyr_dw)):
            if file.endswith(".csv"):
                file_path = os.path.join(solutions_zephyr_dw, file)
                energies, states_dict = read_dwave_solutions(file_path)
                name = file.split(".")[0]

                df = load_instance_to_dataframe(os.path.join(instances_zephyr, name + "_sg.txt"))
                with open(os.path.join(instances_zephyr, name + "_dv.pkl"), "rb") as f:
                    h, J = pickle.load(f)
                matrix, holes = load_instance_to_matrix(os.path.join(instances_zephyr, name + "_sg.txt"))
                biases = deepcopy(matrix.diagonal())
                np.fill_diagonal(matrix, 0)

                # for state in states_dict:
                #     for hole in holes:
                #         state[str(hole)] = 0
                # assert len(states_dict[0]) == len(biases)
                # for hole in holes:
                #     h[hole] = 0
                # for idx, value in enumerate(biases):
                #     try:
                #         assert isclose(h[idx], biases[idx], abs_tol=1e-12)
                #     except:
                #         print(idx)

                h_ordered = dict(sorted(h.items()))
                nodes = df["i"].unique()
                renumeration = {dw: nodes[idx] for idx, dw in enumerate(h.keys())}
                h_renumerated = {renumeration[k]: v for k,v in h.items()}
                for row in df.itertuples():
                    if row.i == row.j:
                        try:
                            assert isclose(row.v, h_renumerated[row.i])
                        except:
                            print(row.i)
                            print(row.v)
                            print(h_renumerated[row.i])



                for idx, state_dict in enumerate(states_dict):
                    energy = energies[idx]

                    state_dict_renumerated = {renumeration[int(k)]: v for k, v in state_dict.items()}
                    en = 0
                    for i, v in h.items():
                        en += state_dict[str(i)] * v
                    for (e1, e2), v in J.items():
                        en += state_dict[str(e1)] * state_dict[str(e2)] * v
                    assert isclose(en, energy)
                    en2 = 0
                    for row in df.itertuples():
                        if row.i == row.j:
                            en2 += row.v * state_dict_renumerated[row.i]
                        else:
                            en2 += state_dict_renumerated[row.i] * state_dict_renumerated[row.j] * row.v
                    assert isclose(en, en2)
                    state_dict_renumerated_ordered = dict(sorted(state_dict_renumerated.items()))
                    state = np.array([i for i in state_dict_renumerated_ordered.values()])
                    for hole in holes:
                        state = np.insert(state, hole, 0)
                    linear = np.dot(state, biases)
                    en3 = 0
                    for row in df.itertuples():
                        if row.i == row.j:
                            en3 += row.v * state_dict_renumerated[row.i]
                    assert isclose(linear, en3)

                    quadratic = np.dot(state.T, np.dot(matrix, state))
                    assert isclose(energy, linear + quadratic)
                    break

                    #
                    #
                    # energy = energies[idx]
                    # en = 0
                    # en2 = 0
                    # for (e1, e2), v in J.items():
                    #     en2 += state_dict[str(e1)] * state_dict[str(e2)] * v


                    # for row in df.itertuples():
                    #     if row.i != row.j:
                    #         en += state_dict_int[row.i-1] * state_dict_int[row.j-1] * row.v
                    # assert isclose(energy, en)

                    #



                # biases_dict = {}
                # for row in df.itertuples():
                #     if row.i == row.j:
                #         biases_dict[row.i] = row.v
                # renumeration = {}
                # # for item, value in h.items():
                # #     for key, row_v in biases_dict.items():
                # #         if isclose(value, row_v):
                # #             renumeration[item] = key
                #
                # for index, state_dict in enumerate(states_dict):
                #     ordered_state = dict(sorted(state_dict.items(), key=lambda item: int(item[0])))
                #     state_array = [i for i in state_dict.values()]
                #
                #     for hole in reversed(holes):
                #         state_array.insert(hole, 0)
                #     state = np.array(state_array)
                #     assert len(biases) == len(state)
                #     linear = np.dot(state, biases)
                #     quadratic = np.dot(state.T, np.dot(matrix, state))
                #     energy = energies[index]
                #     assert isclose == linear + quadratic
            break
