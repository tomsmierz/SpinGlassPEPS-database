
import dwave_networkx as dnx
from typing import Dict


import os

cwd = os.getcwd()


def machine_to_5_tuple(h: Dict) -> Dict:
    h_tuple = {}

    l = sorted(list(h.keys()))
    s = l[0]
    s = dnx.pegasus_coordinates(16).linear_to_nice(s)
    if s[0] > 0:
        raise ValueError("s")
    for node in l:
        h_tuple[node] = dnx.pegasus_coordinates(16).linear_to_nice(node)
        tmp = h_tuple[node]
        t = (tmp[0], tmp[1] - s[1], tmp[2] - s[2], tmp[3], tmp[4])
        bl = [x >= 0 for x in t]
        if not all(bl):
            print(t)
            raise ValueError("t")
        h_tuple[node] = (tmp[0], tmp[1] - s[1], tmp[2] - s[2], tmp[3], tmp[4])

    return h_tuple


def tuple_to_linear(h_tuple: Dict, size: int) -> Dict:
    h_linear = {}
    for value in h_tuple.keys():
        if value[3] == 1:
            x = 4 + value[4] + 1
        else:
            x = abs(value[4] - 3) + 1
        y = abs(value[1] - (size - 2))

        h_linear[value] = 8 * value[0] + 24 * value[2] + 24 * (size - 1) * y + x
        # 24 * (size - 1) * value[0] + 24 * value[1] + 8 * value[2] + 4 * value[3] + value[4] + 1

    return h_linear


def tuple_to_dattani(h_tuple: Dict) -> Dict:
    h_dattani = {}

    for key, value in h_tuple.items():
        tmp = value
        h_dattani[key] = (tmp[2], tmp[1], tmp[0], tmp[3], tmp[4])

    return h_dattani


def dattani_to_linear(h_dattani: Dict, size: int) -> Dict:
    h_linear = {}
    for key, value in h_dattani.items():
        h_linear[key] = (
            24 * (size - 1) * value[0]
            + 24 * value[1]
            + 8 * value[2]
            + 4 * value[3]
            + value[4]
            + 1
        )

    return h_linear


def dattani_to_linear_2(h_dattani: Dict, size: int) -> Dict:
    h_linear = {}
    for key, value in h_dattani.items():
        h_linear[key] = (
            24 * (size - 1) * value[0]
            + 24 * value[2]
            + 8 * value[1]
            + 4 * value[3]
            + value[4]
            + 1
        )

    return h_linear


def nice_to_spin_glass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    return spin_glas_linear


def advantage_6_1_to_spinglass_int(r: int, size: int) -> int:
    if size not in [4, 8, 16]:
        raise NotImplementedError("only work for P4, P8 and P16")
    (t, y, x, u, k) = dnx.pegasus_coordinates(16).linear_to_nice(r)
    t_off = {4: 0, 8: 2, 16: 0}
    y_off = {(4, t): 2 for t in [0, 1, 2]} | {(8, 2): 3, (8, 0): 2, (8, 1): 2} | {(16, t): 0 for t in [0, 1, 2]}
    x_off = {(4, t): 3 for t in [0, 1, 2]} | {(8, 2): 4, (8, 0): 5, (8, 1): 5} | {(16, t): 0 for t in [0, 1, 2]}
    return nice_to_spin_glass(node=((t-t_off[size]) % 3, y-y_off[(size, t)], x-x_off[(size, t)], u, k), size=size)


def advantage_6_1_to_spinglass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    t_off = {4: 0, 8: 2, 16: 0}
    y_off = {(4, t): 2 for t in [0, 1, 2]} | {(8, 2): 3, (8, 0): 2, (8, 1): 2} | {(16, t): 0 for t in [0, 1, 2]}
    x_off = {(4, t): 3 for t in [0, 1, 2]} | {(8, 2): 4, (8, 0): 5, (8, 1): 5} | {(16, t): 0 for t in [0, 1, 2]}
    return nice_to_spin_glass(node=((t-t_off[size]) % 3, y-y_off[(size, t)], x-x_off[(size, t)], u, k), size=size)
