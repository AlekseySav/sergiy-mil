import numpy as np
from src.milp.milp import _split_subdivision, _is_int

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


def test_is_int():
    assert _is_int(4)
    assert _is_int(-4)
    assert _is_int(0)
    assert not _is_int(3.2)
    assert not _is_int(-3.2)
    assert not _is_int(31123.2123121)


def test_split_subdivision():
    t = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 6],
        [1, 0, 0, 7, 8],
        [0, 0, 1, 9, 10]
    ]), basis=[1, 0, 2],
        func=array([3, 4, 5, 0]))
    res = _split_subdivision(t, 0)
    assert len(res) == 2
    t1 = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 0, 6],
        [1, 0, 0, 7, 0, 8],
        [0, 0, 1, 9, 0, 10],
        [0, 0, 0, -7, 1, 0]
    ]), basis=[1, 0, 2, 4],
        func=array([3, 4, 5, 0, 0]))
    t2 = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 0, 6],
        [1, 0, 0, 7, 0, 8],
        [0, 0, 1, 9, 0, 10],
        [0, 0, 0, 7, 1, -1]
    ]), basis=[1, 0, 2, 4],
        func=array([3, 4, 5, 0, 0]))
    for t_new in [t1, t2]:
        assert t_new in res
