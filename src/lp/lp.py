from fractions import Fraction
from dataclasses import dataclass
import operator as op


@dataclass
class Variable:
    is_int: bool
    lower_bound: Fraction | None = None
    upper_bound: Fraction | None =  None
    value: Fraction = Fraction()
    displacement: Fraction = Fraction()


Column = list[Fraction]
Matrix = list[list[Fraction]]


def normalize(constraints: Matrix, bounds: Column, objectives: Column, vars: list[Variable]) -> None:
    assert len(objectives) == len(constraints[0])
    n = len(vars)
    m = len(objectives)

    # make variables positive, unbounded
    def add_bound(bound, op, a, b):
        if bound is None:
            return
        if op(bound, 0):
            vars[a].displacement = bound
            constraints.append([0 if i != b else 1 for i in range(len(vars))])
            bounds.append(0)
        else:
            constraints.append([0 if i != b else 1 for i in range(len(vars))])
            bounds.append(-bound)

    for i in range(n):
        # x_{i} - x_{i+n} is original x_{i}
        vars.append(Variable(vars[i].is_int, 0))
        v = vars[i]
        add_bound(v.lower_bound, op.gt, i, n + i)
        add_bound(v.upper_bound, op.lt, i, n + i)
        v.upper_bound = None
        v.lower_bound = 0

    # constraints -> equations
    equ = []
    for id, line in enumerate(constraints):
        equ.append([
                line[i] if i < len(line) else
                -line[i - m] if i < 2 * m else
                0 if i != id + 2 * m else
                1
                for i in range(2 * m + len(constraints))
            ]
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


def nice_print(M, B, O, V):
    for m, b in zip(M, B):
        print(' + '.join(f'{i}*x{ii}' for ii, i in enumerate(m)), f'<= {b}')
    print('max: ', ' + '.join(f'{i}*x{ii}' for ii, i in enumerate(O)))
    print('where: ', ', '.join(f'{v.lower_bound} <= x{i}-{v.displacement}({"int" if v.is_int else "real"}) <= {v.upper_bound}'
        for i, v in enumerate(V)))

# M = [
#     [1, 2, 3]
# ]

# B = [6]

# O = [1, 9, 0]

# V = [Variable(False, 2, 5), Variable(False), Variable(True)]

# nice_print(M, B, O, V)
# normalize(M, B, O, V)
# nice_print(M, B, O, V)
