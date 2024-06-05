import dwave_networkx as dnx
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
instances_base = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
dw_pegasus_files = os.path.join(ROOT, "data", "pegasus", "dwave")
tn_pegasus_files = os.path.join(ROOT, "data", "pegasus", "tn")
dw_zephyr_files = os.path.join(ROOT, "data", "zephyr", "dwave")
tn_zephyr_files = os.path.join(ROOT, "data", "zephyr", "tn")

path_to_solutions_base = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies"
size = "Z3"
instances_class = "RCO"
truncation = "truncate2^16"

solutions_pegasus_tn = os.path.join(path_to_solutions_base, "pegasus_random_tn", size, instances_class, f"final_bench_{truncation}")
instances_pegasus = os.path.join(instances_base, "pegasus_random", size, instances_class)

solutions_zephyr_dw = os.path.join(path_to_solutions_base, "aggregated", "zephyr_random", size, instances_class)


instances_zephyr = os.path.join(instances_base, "zephyr_random", size, instances_class)

save_pegasus_tn = os.path.join(ROOT, "data", "pegasus", "tn", size, truncation, instances_class)
save_zephyr_dw = os.path.join(ROOT, "data", "zephyr", "dwave", size, instances_class)


if __name__ == '__main__':
    name = "001"
    with open(os.path.join(instances_zephyr, name + "_dv.pkl"), "rb") as f:
        h, J = pickle.load(f)

    z4 = dnx.zephyr_graph(4)
    plt.figure(figsize=(20, 20))
    dnx.draw_zephyr(z4, with_labels=True, edgelist=list(J.keys()))
    #
    plt.savefig("tmp.pdf")