import os

import numpy as np

from clean_start import read_instance_states_energies_from_tn, load_instance_to_matrix, load_instance_to_dataframe
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
instances_base = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
dw_pegasus_files = os.path.join(ROOT, "data", "pegasus", "dwave")
tn_pegasus_files = os.path.join(ROOT, "data", "pegasus", "tn")
dw_zephyr_files = os.path.join(ROOT, "data", "zephyr", "dwave")
tn_zephyr_files = os.path.join(ROOT, "data", "zephyr", "tn")

path_to_solutions_base = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies"
size = "P8"
instances_class = "CBFM-P"
truncation = "truncate2^16"

solutions_pegasus_tn = os.path.join(path_to_solutions_base, "pegasus_random_tn", size, instances_class, f"final_bench_{truncation}")
instances_pegasus = os.path.join(instances_base, "pegasus_random", size, instances_class)

solutions_zephyr_dw = os.path.join(path_to_solutions_base, "aggregated", "zephyr_random", size, instances_class)


instances_zephyr = os.path.join(instances_base, "zephyr_random", size, instances_class)

save_pegasus_tn = os.path.join(ROOT, "data", "pegasus", "tn", size, truncation, instances_class)
save_zephyr_dw = os.path.join(ROOT, "data", "zephyr", "dwave", size, instances_class)

if __name__ == '__main__':

    for file in os.listdir(solutions_pegasus_tn):
        if file.endswith(".json"):
            file_path = os.path.join(solutions_pegasus_tn, file)
            name, energies, states, dict_of_states = read_instance_states_energies_from_tn(file_path)
            df = load_instance_to_dataframe(os.path.join(instances_pegasus, name + "_sg.txt"))
            matrix = load_instance_to_matrix(os.path.join(instances_pegasus, name + "_sg.txt"))
            biases = deepcopy(matrix.diagonal())
            np.fill_diagonal(matrix, 0)

            for index, state_dict in enumerate(dict_of_states):
                state = states[index, :]
                for key, value in state_dict.items():
                    assert state[int(key)-1] == value
                en = 0
                for row in df.itertuples():
                    if row.i == row.j:
                        en += state_dict[str(row.i)] * row.v
                    else:
                        en += state_dict[str(row.i)] * state_dict[str(row.j)] * row.v
                assert en == energies[index]


            for index, energy in enumerate(energies):
                state = states[index, :]
                linear = np.dot(state, biases)
                quadratic = np.dot(state.T, np.dot(matrix, state))
                assert energy == linear + quadratic
