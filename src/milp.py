from copy import deepcopy
from time import time

import numpy as np

import lp
from lp import Tableau, NDArray, Float, make_solution_optimal, make_solution_feasible
from heuristic import func_get_tableau, func_get_axis

eps = 1e-5


def _is_int(x, delta=eps):
    return abs(x - round(x)) <= delta


def split_subdivision(problem: Tableau, var_index: int) -> list[Tableau]:
    value = problem.solution()[1][var_index]

    new_problem = deepcopy(problem)

    new_problem.add_restriction(var_index, np.floor(value), lp.ConstraintSign.LEQ)
    problem.add_restriction(var_index, np.floor(value)+1, lp.ConstraintSign.GEQ)

    return [new_problem, problem]


def check_solution(values: NDArray, constraints: list[bool]) -> list[bool]:
    return [not constraint or _is_int(value)
            for constraint, value in zip(constraints, values)]


def solve_milp(problem: Tableau, constraints: list[bool],
               get_tableau: func_get_tableau,
               get_axis: func_get_axis,
               it_limit: int = 1000,
               bb_limit: int = 1000,
               solution: list[float] | None = None) -> tuple[float | None, str]:
    """
    Solves Mixed Integer Linear maximization Problem in Tableau form.
    """
    start = time()
    output = "\nMILP output:\n"

    iteration = 0
    bb_nodes = 0
    if not make_solution_optimal(problem):
        output += 'The problem does not have an optimal solution.\n'
        return None, output
    z_upper, best_solution = problem.solution()
    z_lower = Float(0)
    subdivisions = [problem]
    bb_nodes += 1
    while subdivisions:
        print(iteration)
        iteration += 1
        if iteration > it_limit:
            return None, output + f'Iterations limit ({it_limit} exceeded\n)'
        problem = get_tableau(subdivisions, constraints)
        subdivisions.remove(problem)
        if not make_solution_feasible(problem):
            # infeasible
            continue
        z, values = problem.solution()
        z_upper = max(z_upper, z)
        if z <= z_lower:
            continue
        in_constraints = check_solution(values, constraints)
        if all(in_constraints):
            z_lower = z
            best_solution = values
            if z_lower == z_upper:
                break
            else:
                continue
        else:
            split_index = get_axis(problem, in_constraints)
            if split_index is None:
                continue
            subdivisions += split_subdivision(problem, split_index)
            bb_nodes += 2
            if bb_nodes > bb_limit:
                return None, output + f'Branch&bounds nodes limit ({bb_limit} exceeded)\n'

    end = time()
    output += f"solution: \n {best_solution[:len(constraints)]}\n"
    output += f"solution value = {z_lower}\n"
    output += f"Problem solved in {iteration} iterations\n"
    output += f"Problem solved in {bb_nodes} branch&bound nodes\n"
    output += f"Problem solved in {end - start} seconds\n"
    if solution is not None:
        for i in best_solution[:len(constraints)]:
            solution.append(i)
    return z_lower, output
