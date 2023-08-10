import math
from copy import deepcopy

import numpy as np

from src.tableau import Tableau, array
from src.lp import solve_lp, make_solution_feasible
from src.heuristic import func_get_tableau, func_get_axis

eps = 1/1000000000


def _is_int(x, delta=eps):
    return abs(x - round(x)) <= delta


def _subdivision(problem: Tableau, var_index: int, less: bool) -> Tableau:
    row_index = problem.basis.index(var_index)
    value = math.floor(problem.matrix[row_index, -1])

    new_problem = deepcopy(problem)
    if value == 0 and less:
        value += eps
    constraint = np.zeros(problem.variables_count + 1, dtype=np.float64)
    constraint[var_index] = 1 if less else -1
    constraint[-1] = value if less else -value - 1
    constraint = constraint - problem.matrix[row_index] if less else constraint + problem.matrix[row_index]

    new_problem.remember_constraint(var_index, value + 1 if not less else value, less)
    new_problem.add_constraint(constraint)
    return new_problem


def _split_subdivision(problem: Tableau, var_index: int) -> list[Tableau]:
    return [_subdivision(problem, var_index, True),
            _subdivision(problem, var_index, False)]


def _get_solution(problem: Tableau, constraints: list[bool]) -> list[float]:
    # first n columns in Tableau's matrix correspond to initial variables
    n = len(constraints)
    solution = [0.0 for _ in range(n)]
    for row, x in enumerate(problem.basis):
        if x < n:
            solution[x] = problem.matrix[row, -1]
    return solution


def _check_solution(problem: Tableau, constraints: list[bool]) -> list[bool]:
    solution = _get_solution(problem, constraints)
    return [not constraint or _is_int(value)
            for constraint, value in zip(constraints, solution)]


def _solution(problem: Tableau, constraints: list[bool]) -> float:
    solution = _get_solution(problem, constraints)
    in_constraints = _check_solution(problem, constraints)
    if all(in_constraints):
        for i, x in enumerate(solution):
            if constraints[i]:
                solution[i] = round(x)
    return array(solution) @ problem.func[:len(constraints)]


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
    if not solve_lp(problem):
        return None
    z_lower = problem.solution()
    z_upper = 0
    subdivisions = [problem]
    while subdivisions:
        iteration += 1
        problem = get_tableau(subdivisions, constraints)
        subdivisions.remove(problem)
        if not make_solution_feasible(problem):
            # infeasible
            continue
        if not solve_lp(problem):
            # infeasible
            continue
        z = problem.solution()
        # print(problem.variables_constraints)
        # print(z)
        # print(z_upper)
        z_lower = min(z_lower, z)
        if z >= z_upper:
            continue
        in_constraints = _check_solution(problem, constraints)
        if all(in_constraints):
            print()
            print('iteration = ', iteration)
            print('new best solution = ', _get_solution(problem, constraints))
            z_upper = _solution(problem, constraints)
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
            subdivisions += _split_subdivision(problem, split_index)

    return z_upper
