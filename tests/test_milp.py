import numpy as np
from src.milp.milp import _split_subdivision

from tableau import Tableau, array

'''
Problem:
    Min/Max 7x+6y, where
        2x + 4y <= 16
        3x + 2y <= 12
        x, y >= 0

Setup:
    t = Tableau(func=array([7, 6]))
    t.add_constraint(array([2, 4, 16]))            # 2x + 4y <= 16
    t.add_constraint(array([3, 2, 0, 12]))         # 3x + 2y <= 12
                            ^~~~~~~~ s1
Result after setup:
    func        7  6  0  0
    matrix      2  4  1  0  16
                3  2  0  1  12
    basis             ^  ^
    count=4
'''


def test_split_subdivision():
    t = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 6],
        [1, 0, 0, 7, 8],
        [0, 0, 1, 9, 10]
    ]), basis=[0, 1, 2],
        func=array([3, 4, 5, 0]))
    res = _split_subdivision(t, 0)
    assert len(res) == 2
