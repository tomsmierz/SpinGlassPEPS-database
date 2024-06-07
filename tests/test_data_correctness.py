import unittest
import os
import pickle
from data_creation.clean_start import load_instance_to_dataframe, read_dwave_solutions


class Zephyr4RCO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path_to_solutions_base = rf"C:\Users\walle\PycharmProjects\D-Wave_Scripts\energies"
        instances_base = r"C:\Users\walle\PycharmProjects\D-Wave_Scripts\instances"
        instances_zephyr = os.path.join(instances_base, "zephyr_random", "Z4", "RCO")
        solutions_zephyr_dw = os.path.join(path_to_solutions_base, "aggregated", "zephyr_random", "Z4", "RCO")
        name = "001"
        with open(os.path.join(instances_zephyr, name + "_dv.pkl"), "rb") as f:
            cls.h, cls.J = pickle.load(f)
        cls.df = load_instance_to_dataframe(os.path.join(instances_zephyr, name + "_sg.txt"))
        cls.energies, cls.states_dict = read_dwave_solutions(os.path.join(solutions_zephyr_dw, name + ".csv"))

    def test_using_df(self):
        best_solution = self.states_dict[0]
        best_energy = self.energies[0]
        en = 0
        for node, value in self.h.items():
            en += best_solution[str(node)] * value
        for (e1, e2), value in self.J.items():
            en += best_solution[str(e1)] * best_solution[str(e2)] * value
        self.assertAlmostEqual(en, best_energy)


if __name__ == '__main__':
    unittest.main()
