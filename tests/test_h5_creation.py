import unittest
import os
import h5py
import pandas as pd
from data_creation.aggregate_data import create_h5_file_from_dwave

ROOT = root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDWaveCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        instance_name = "test_dwave.txt"
        data_name = "test_dwave.csv"
        file_name = "test_dwave.hdf5"
        path = os.path.join(ROOT, "tests", "test_data")
        create_h5_file_from_dwave(path, path, path, instance_name, data_name,
                                  file_name, 4)
        cls.read_file = os.path.join(path, file_name)
        cls.read_instance = os.path.join(path, instance_name)

    def test_save_quality(self):
        with h5py.File(self.read_file, "r") as f:
            J_coo = f["Ising"]["J_coo"]
            biases = f["Ising"]["biases"]
            spectrum = f["Spectrum"]
            dtype_spec = {
                'i': 'int',
                'j': 'int',
                'v': 'float64'
            }
            pd.set_option('display.precision', 20)
            instance_df = pd.read_csv(self.read_instance, sep=" ", index_col=False,
                                      header=None, comment='#', names=["i", "j", "v"], dtype=dtype_spec)
            for row in instance_df.itertuples():
                if row.i == row.j:
                    self.assertAlmostEqual(row.v, biases[row.i-1], places=7)
            I = J_coo["I"]
            J = J_coo["J"]
            V = J_coo["V"]
            for index in range(len(I)):
                i, j, v = I[index], J[index], V[index]
                df_value = instance_df.loc[(instance_df['i'] == i) & (instance_df["j"] == j), 'v'].values[0]
                self.assertAlmostEqual(df_value, v, places=7)
            states = spectrum["states"]
            energies = spectrum["energies"]
            for i in range(len(energies)):
                state = states[:, i]
                en = 0
                for index, bias in enumerate(biases):
                    en += bias * state[index]
                for index in range(len(I)):
                    en += V[index] * state[I[index]-1] * state[J[index]-1]

                self.assertAlmostEqual(en, energies[i], places=6)



if __name__ == '__main__':
    unittest.main()
