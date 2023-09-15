from milp import solve_milp, eps, _is_int
from heuristic import gt_simple, ga_simple, gt_min, gt_max
from lp import Tableau, array
from or_tools_api import solve_or_tools
from problem import Problem
from problem_gen import ProblemGenerator

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


def test_milp_simple():
    p = Problem(
        [
            [-2, 2],
            [2, 3]
        ],
        [10, 20],
        [1, 30]
    )
    t = p.to_tableau()
    constraints = [True, True]
    res = solve_milp(t, constraints=constraints, get_tableau=gt_simple, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert res == or_tools_res, print(or_tools_output)


def test_milp_simple():
    p = Problem(
        [[5, -10], [4, 6]],
        [157, 33],
        [6, 9]
    )
    t = p.to_tableau()
    constraints = [True, True]
    res = solve_milp(t, constraints=constraints, get_tableau=gt_simple, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert res == or_tools_res, print(or_tools_output)

def test_milp_simple():
    p = Problem(
        [[5, -10], [4, 6], [0, 1]],
        [157, 33, 5],
        [6, 9]
    )
    t = p.to_tableau()
    constraints = [True, True]
    res = solve_milp(t, constraints=constraints, get_tableau=gt_simple, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert res == or_tools_res, print(or_tools_output)


def test_with_or_tools():
    p = Problem([[1, 1], [5, 9]], [6, 45], [5, 8])
    t = p.to_tableau()
    constraints = [True, True]
    res = solve_milp(t, constraints=constraints, get_tableau=gt_simple, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert res == or_tools_res, print(or_tools_output)


def test_from_or_tools_example():
    p = Problem(
        [
            [5, 7, 9, 2, 1],
            [18, 4, -9, 10, 12],
            [4, 7, 3, 8, 5],
            [5, 13, 16, 3, -7],
        ],
        [250, 285, 211, 315],
        [7, 8, 2, 9, 6]
    )
    t = p.to_tableau()
    constraints = [True, True, True, True, True]
    my_res = solve_milp(t, constraints=constraints, get_tableau=gt_simple, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert abs(my_res - or_tools_res) < eps


def test_with_gt_min():
    p = Problem(
        [
            [5, 7, 9, 2, 1],
            [18, 4, -9, 10, 12],
            [4, 7, 3, 8, 5],
            [5, 13, 16, 3, -7],
        ],
        [250, 285, 211, 315],
        [7, 8, 2, 9, 6]
    )
    t = p.to_tableau()
    constraints = [True, True, True, True, True]
    my_res = solve_milp(t, constraints=constraints, get_tableau=gt_min, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert abs(my_res - or_tools_res) < eps


def test_with_gt_max():
    p = Problem(
        [
            [5, 7, 9, 2, 1],
            [18, 4, -9, 10, 12],
            [4, 7, 3, 8, 5],
            [5, 13, 16, 3, -7],
        ],
        [250, 285, 211, 315],
        [7, 8, 2, 9, 6]
    )
    t = p.to_tableau()
    constraints = [True, True, True, True, True]
    my_res = solve_milp(t, constraints=constraints, get_tableau=gt_max, get_axis=ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    assert abs(my_res - or_tools_res) < eps


def test_small_random():
    gen = ProblemGenerator(
        {
            'constraints_coeffs': 'uniform',
            'bounds': 'uniform',
            'obj_coeffs': 'uniform'
        },
        {
            'constraints_coeffs': [-10, 10],
            'bounds': [0, 200],
            'obj_coeffs': [-10, 10]
        },
        2,
        2
    )
    p = gen.value()
    t = p.to_tableau()

    cons = [True for _ in range(2)]
    my_res = solve_milp(t, cons, gt_min, ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    if any([my_res is not None, or_tools_res is not None]):
        assert abs(my_res - or_tools_res) < eps, print(p.constraint_coeffs, p.bounds, p.obj_coeffs)


def test_mid_random():
    gen = ProblemGenerator(
        {
            'constraints_coeffs': 'uniform',
            'bounds': 'uniform',
            'obj_coeffs': 'uniform'
        },
        {
            'constraints_coeffs': [-10, 10],
            'bounds': [0, 200],
            'obj_coeffs': [-10, 10]
        },
        10,
        10
    )
    p = gen.value()
    t = p.to_tableau()
    cons = [True for _ in range(10)]
    my_res = solve_milp(t, cons, gt_max, ga_simple)
    or_tools_res, or_tools_output = solve_or_tools(p)
    if any([my_res is not None, or_tools_res is not None]):
        assert abs(my_res - or_tools_res) < eps, print(p.constraint_coeffs, p.bounds, p.obj_coeffs)


def test_many_small_random():
    for i in range(100):
        print(i)
        test_small_random()


def test_many_mid_random():
    for i in range(100):
        print(i)
        test_mid_random()
