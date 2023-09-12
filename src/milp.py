import math
from copy import deepcopy

import numpy as np

import lp
from src.lp import Tableau, NDArray, Float, array, make_solution_optimal, make_solution_feasible
from src.heuristic import func_get_tableau, func_get_axis

eps = 1 / 1000000000


def _is_int(x, delta=eps):
    return abs(x - round(x)) <= delta


def split_subdivision(problem: Tableau, var_index: int) -> list[Tableau]:
    value = problem.solution()[1][var_index]

    new_problem = deepcopy(problem)

    new_problem.add_restriction(var_index, np.round(value), lp.ConstraintSign.LEQ)
    problem.add_restriction(var_index, np.round(value)+1, lp.ConstraintSign.GEQ)

    return [new_problem, problem]


def check_solution(values: NDArray, constraints: list[bool]) -> list[bool]:
    return [not constraint or _is_int(value)
            for constraint, value in zip(constraints, values)]


def solve_milp(problem: Tableau, constraints: list[bool],
               get_tableau: func_get_tableau,
               get_axis: func_get_axis) -> float | None:
    """

    Solves Mixed Integer Linear minimization Problem in Tableau form.

    #### designations:
        - A = problem.matrix
        - basis = problem.basis

    #### requirements:
        - problem should be feasible, i.e.
        > basis should correspond to s_i variables
        > A[:,basis] should be E ( matrix, with main diagonal set to ones)
        > last column of A >= 0

    """
    iteration = 0
    if not make_solution_optimal(problem):
        return None
    z_lower, _ = problem.solution()
    z_upper = Float(0)
    subdivisions = [problem]
    while subdivisions:
        iteration += 1
        problem = get_tableau(subdivisions, constraints)
        subdivisions.remove(problem)
        if not make_solution_feasible(problem):
            # infeasible
            continue
        z, values = problem.solution()
        # print(problem.variables_constraints)
        # print(z)
        # print(z_upper)
        z_lower = min(z_lower, z)
        if z >= z_upper:
            continue
        in_constraints = check_solution(values, constraints)
        if all(in_constraints):
            print()
            print('iteration = ', iteration)
            print('new best solution = ', values[:len(constraints)])
            z_upper = z # TODO : Check
            print('solution value (floored small variables)= ', z_upper)
            if z_lower == z_upper:
                break
            else:
                continue
        else:
            split_index = get_axis(problem, in_constraints)
            if split_index is None:
                continue
            # print('split, index = ', split_index)
            subdivisions += split_subdivision(problem, split_index)

    return z_upper
