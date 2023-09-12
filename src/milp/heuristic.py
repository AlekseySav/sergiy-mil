from collections.abc import Callable

from numpy import argmax

from lp import solve_lp
from src.tableau import Tableau, array

func_get_tableau = Callable[[list[Tableau], list[bool]], Tableau | None]
func_get_axis = Callable[[Tableau, list[bool]], int | None]


def gt_simple(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
    if problems:
        return problems[0]
    return None


def ga_simple(problem: Tableau, in_constraints: list[bool]) -> int | None:
    for i, val in enumerate(in_constraints):
        if not val:
            return i
    return None


def gt_fifo(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
    if problems:
        return problems[-1]
    return None


def gt_max_solution(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
    if not problems:
        return None
    z = [p.solution() if solve_lp(p) else float('-inf') for p in problems]
    return problems[argmax(array(z))]
