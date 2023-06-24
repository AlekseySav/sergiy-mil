from fractions import Fraction
from dataclasses import field, dataclass
import operator as op
import numpy as np


Number = np.float32


@dataclass
class Variable:
    is_int: bool
    lower_bound: Number | None = None
    upper_bound: Number | None =  None
    displacement: Number = 0
    # convert to original variable
    negate: bool = False
    negative_part: int | None = None
    slack: bool = True


@dataclass
class Solver:
    # minimize cx, where Ax=b
    objectives: np.ndarray[Number] = field(default_factory=lambda: np.array([]))  # c
    bounds: np.ndarray[Number] = field(default_factory=lambda: np.array([]))  # b
    variables: list[Variable] = field(default_factory=list)
    equations: np.ndarray[Number] = field(default_factory=lambda: np.array([]))  # A (matrix actually)
    solution: np.ndarray[Number] = field(default_factory=lambda: np.array([]))  # x


    def add_var(self, v: Variable) -> None:
        assert self.equations.size == 0
        v.slack = False
        self.variables.append(v)


    def add_constraint(self, constraint: np.ndarray[Number], bound: Number, negate: bool = True) -> None:
        v = 0 if len(self.equations.shape) < 2 else self.equations.shape[0]
        self.variables.append(Variable(False, 0, None))
        if self.equations.size == 0:
            self.equations = np.zeros(shape=(1, len(self.variables)))
        else:
            self.equations = np.append(self.equations, np.zeros(shape=(1, self.equations.shape[1])), axis=0)
            self.equations = np.append(self.equations, np.zeros(shape=(self.equations.shape[0], 1)), axis=1)

        for i in range(len(self.variables) - len(constraint)):
            constraint = np.append(constraint, [0])
        for i in range(len(constraint)):
            if not negate:
                continue
            if self.variables[i].negative_part:
                constraint[self.variables[i].negative_part] = constraint[i]
        for i in range(len(constraint)):
            if not negate:
                continue
            if self.variables[i].negate:
                constraint[i] = -constraint[i]
 
        self.equations[v,:len(constraint)] = constraint
        self.equations[v][len(self.variables) - 1] = 1
        self.bounds = np.append(self.bounds, bound)


    def normalize(self) -> None:
        m = len(self.objectives)
        n = len(self.equations)

        # remove negative variables
        for i in range(m):
            v = self.variables[i]
            # (0) lower bound > 0 => only positive part
            # (1) upper bound < 0 => only negative part
            if v.upper_bound and v.upper_bound < 0:
                v.negate = True
                v.upper_bound, v.lower_bound = -v.lower_bound, -v.upper_bound
                continue
            # (2) lower bound < 0, upper bound > 0 => split into negative & positive parts
            if not v.lower_bound or v.lower_bound < 0:
                self.variables.append(Variable(v.is_int, 0, v.lower_bound and -v.lower_bound, 0, True))
                v.lower_bound = 0
                v.negative_part = len(self.variables) - 1
                if self.equations.size != 0:
                    self.equations = np.append(self.equations, self.equations[:,i].reshape(n, 1), axis=1)
                self.objectives = np.append(self.objectives, self.objectives[i])

        # apply negate flag
        for i, v in enumerate(self.variables):
            if v.negate:
                self.objectives[i] = -self.objectives[i]

        # shift bounds
        for i, v in enumerate(self.variables):
            v.displacement = v.lower_bound
            if v.upper_bound:
                v.upper_bound -= v.displacement
                self.add_constraint(np.array([int(i == j) for j in range(len(self.variables))]), v.upper_bound, False)
                v.upper_bound = None
            v.lower_bound = 0

        self.solution.resize((len(self.variables),))


    def evalutate(self) -> list[Number]:
        res = []
        def get(v: Variable, n: Number):
            if v.negate:
                n = -n
            n += v.displacement
            return n
        for i, v in enumerate(self.variables):
            if v.slack:
                continue
            p = v.negative_part
            res.append(get(v, self.solution[i]) + (get(self.variables[p], self.solution[p]) if p else 0))
        return res


def nice_print(s):
    M = s.equations
    B = s.bounds
    O = s.objectives
    V = s.variables
    for m, b in zip(M, B):
        print(' + '.join(f'{i}*{"s" if v.slack else "x"}{ii}' for ii, (i, v) in enumerate(zip(m, V))), f'= {b}')
    print('max: ', ' + '.join(f'{i}*x{ii}' for ii, i in enumerate(O)))
    print('where: ', ', '.join(f'{v.lower_bound} <= {"neg " if v.negate else ""}{"s" if v.slack else "x"}{i}-{v.displacement}({"int" if v.is_int else "real"}) <= {v.upper_bound}'
        for i, v in enumerate(V)))


s = Solver()

s.add_var(Variable(False, 2, 5))
s.add_var(Variable(False, -3, -2))
s.add_var(Variable(False, -8, 9))

s.objectives = np.array([1, 9, 0], dtype=Number)

# nice_print(s)
s.normalize()

s.add_constraint(np.array([1, 2, 3], dtype=Number), 6)
s.add_constraint(np.array([1, 3, 7], dtype=Number), 3)

nice_print(s)
