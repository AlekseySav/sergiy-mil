import numpy as np
from src.milp.milp import _split_subdivision, _is_int, _get_solution, _check_solution, solve_milp
from src.milp.heuristic import gt_simple, ga_simple
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


def test_split_subdivision_simple():
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


def test_split_subdivision_non_integer():
    t = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 6.001],
        [1, 0, 0, 7, 8.001],
        [0, 0, 1, 9, 10]
    ]), basis=[1, 0, 2],
        func=array([3, 4, 5, 0]))
    res = _split_subdivision(t, 0)
    assert len(res) == 2
    t1 = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 0, 6.001],
        [1, 0, 0, 7, 0, 8.001],
        [0, 0, 1, 9, 0, 10],
        [0, 0, 0, -7, 1, -0.001]
    ]), basis=[1, 0, 2, 4],
        func=array([3, 4, 5, 0, 0]))
    t2 = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 0, 6.001],
        [1, 0, 0, 7, 0, 8.001],
        [0, 0, 1, 9, 0, 10],
        [0, 0, 0, 7, 1, -0.999]
    ]), basis=[1, 0, 2, 4],
        func=array([3, 4, 5, 0, 0]))
    for t_new in [t1, t2]:
        assert t_new in res


def test_get_solution():
    t = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 6],
        [1, 0, 0, 7, 8],
        [0, 0, 1, 9, 10]
    ]), basis=[1, 0, 2],
        func=array([3, 4, 5, 0]))
    res = _get_solution(t, [False, False, False])
    assert len(res) == 3
    assert res == [8, 6, 10]


def test_check_solution_correct_case():
    t = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 6.5],
        [1, 0, 0, 7, 8],
        [0, 0, 1, 9, 10]
    ]), basis=[1, 0, 2],
        func=array([3, 4, 5, 0]))
    res = _check_solution(t, [True, False, True])
    assert len(res) == 3
    assert all(res)


def test_check_solution_wrong_case():
    t = Tableau.from_matrix(matrix=array([
        [0, 1, 0, 5, 6],
        [1, 0, 0, 7, 8.5],
        [0, 0, 1, 9, 10]
    ]), basis=[1, 0, 2],
        func=array([3, 4, 5, 0]))
    res = _check_solution(t, [True, False, True])
    assert len(res) == 3
    assert not all(res)


def test_milp_simple():
    t = Tableau(func=array([-5, -8]))
    t.add_constraint(array([1, 1, 6]))
    t.add_constraint(array([5, 9, 0, 45]))
    t.add_constraint(array([-1, 0, 0, 0, 0]))
    t.add_constraint(array([0, -1, 0, 0, 0, 0]))
    t.print()
    constraints = [True, True]
    res = solve_milp(t, constraints=constraints, get_tableau=gt_simple, get_axis=ga_simple)
    assert res == 40