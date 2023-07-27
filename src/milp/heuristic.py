from collections.abc import Callable

from src.tableau import Tableau


func_get_tableau: Callable[[list[Tableau], list[bool]], Tableau | None]
func_get_axis: Callable[[Tableau, list[bool]], int | None]


def gt_simple(problems: list[Tableau], constraints: list[bool]) -> Tableau | None:
    if len(problems) > 0:
        return problems[0]
    return None


def ga_simple(problems: list[Tableau], in_constraints: list[bool]) -> int | None:
    for i, val in enumerate(in_constraints):
        if not val:
            return i
    return None
