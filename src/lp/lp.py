from fractions import Fraction
from dataclasses import dataclass
from src.tableau import Tableau

@dataclass
class Variable:
    is_int: bool
    lower_bound: Fraction | None = None
    upper_bound: Fraction | None =  None
    value: Fraction = Fraction()
    displacement: Fraction = Fraction()


Column = list[Fraction]
Matrix = list[list[Fraction]]


def normalize(constraints: Matrix, objectives: Column, vars: list[Variable]) -> None:
    assert len(objectives) == len(constraints[0])

    n = len(vars)
    m = len(objectives)

    # fix bounds
    for i in range(n):
        # x_{i} - x_{i+n} is original x_{i}
        vars.append(Variable(vars[i].is_int))
        pos, neg = vars[i], vars[i + n]
        if pos.lower_bound is not None:
            pos.displacement = pos.lower_bound
        if pos.upper_bound is not None:
            neg.displacement = -pos.upper_bound
        pos.lower_bound = pos.upper_bound = None

    # constraints -> equations
    equ = []
    for i, line in enumerate(constraints):
        equ.append([
                line[i] if i < m else
                -line[i - m] if i < 2 * m else
                0
                for i in range(len(vars))] + [0]
            )
        vars.append(Variable(False))
    constraints[:] = equ[:]

    # objective coeffs
    for i in range(m):
        objectives.append(-objectives[i])


def evaluate(vars: list[Variable], n: int) -> list[Fraction]:
    res = []
    for i in range(n):
        pos, neg = vars[i], vars[i + n]
        res.append(pos.value + pos.displacement - neg.value - neg.displacement)
    return res

def solve_lp(problem: Tableau):
    ...
