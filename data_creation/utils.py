import h5py
import numpy as np


def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def array_from_dict(dict_list: list[dict]) -> np.ndarray:
    # Determine the number of states and maximum index to determine the size of the resulting array
    num_states = len(dict_list)
    max_index = max(int(key) for d in dict_list for key in d.keys())
    result_array = np.zeros((num_states, max_index))
    for i, d in enumerate(dict_list):
        for key, value in d.items():
            result_array[i, int(key) - 1] = value
    return result_array

