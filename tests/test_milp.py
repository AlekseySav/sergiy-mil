import numpy as np

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

np.set_printoptions(suppress=True)


def comp(res1, res2) -> bool:

    return ((res1 is None) == (res2 is None)) and (res2 - res1 < eps)


def comp_with_or_tools(p: Problem, gt=gt_simple, ga=ga_simple):
    constraints = [True for _ in range(p.num_vars)]
    p.type_constraints = constraints

    t = p.to_tableau()
    milp_res, milp_output = solve_milp(t, constraints, gt, ga)
    or_tools_res, or_tools_output = solve_or_tools(p)

    if any([milp_res is not None, or_tools_res is not None]):
        log = str(p) + "\n\n" + milp_output + "\n\n" + or_tools_output
        assert comp(milp_res, or_tools_res), log


def test_milp_simple1():
    p = Problem(
        [
            [-2, 2],
            [2, 3]
        ],
        [10, 20],
        [1, 30]
    )
    comp_with_or_tools(p)


def test_milp_simple2():
    p = Problem(
        [[5, -10], [4, 6]],
        [157, 33],
        [6, 9]
    )
    comp_with_or_tools(p)


def test_milp_simple3():
    p = Problem(
        [[5, -10], [4, 6], [0, 1]],
        [157, 33, 5],
        [6, 9]
    )
    comp_with_or_tools(p)


def test_milp_simple4():
    p = Problem(
        [[-3, 9], [6, -9]],
        [8, 142],
        [7, -8]
    )
    comp_with_or_tools(p)


def test_with_or_tools():
    p = Problem(
        [[1, 1], [5, 9]],
        [6, 45],
        [5, 8]
    )
    comp_with_or_tools(p)


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
    comp_with_or_tools(p)


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
    comp_with_or_tools(p, gt_min)


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
    comp_with_or_tools(p, gt_max)


def stress_comp(gen: ProblemGenerator, runs: int = 1, gt=gt_simple, ga=ga_simple, milp_iterations_limit: int = 100000,
                milp_branch_bound_limit: int = 100000, ):
    constraints = [True for _ in range(gen.vars)]
    for _ in range(runs):
        p = gen.value()
        p.type_constraints = constraints
        t = p.to_tableau()
        milp_res, milp_output = solve_milp(t, constraints, gt, ga, milp_iterations_limit, milp_branch_bound_limit)
        or_tools_res, or_tools_output = solve_or_tools(p)

        if any([milp_res is not None, or_tools_res is not None]):
            assert comp(milp_res, or_tools_res), str(p) + "\n\n" + milp_output + "\n\n" + or_tools_output


def test_random_2x2():
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
    stress_comp(gen, 100)

