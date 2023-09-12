from collections.abc import Callable

from numpy import argmax, argmin

from src.lp import make_solution_feasible, solve_lp
from src.tableau import Tableau


func_get_tableau = Callable[[list[Tableau], list[bool]], Tableau | None]
func_get_axis = Callable[[Tableau, list[bool]], int | None]


def gt_simple(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
    if len(problems) > 0:
        return problems[0]
    return None


def ga_simple(problem: Tableau, in_constraints: list[bool]) -> int | None:
    for i, val in enumerate(in_constraints):
        if not val:
            return i
    return None


def gt_min(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
    if not problems:
        return None
    for p in problems:
        if not make_solution_feasible(p):
            # infeasible
            continue
        if not solve_lp(p):
            # infeasible
            continue
    values = [p.solution() for p in problems]
    return problems[argmin(values)]


def gt_max(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
    if not problems:
        return None
    for p in problems:
        if not make_solution_feasible(p):
            # infeasible
            continue
        if not solve_lp(p):
            # infeasible
            continue
    values = [p.solution() for p in problems]
    return problems[argmax(values)]
