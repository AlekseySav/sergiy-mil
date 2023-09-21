'''

simplex method implementation in tabular form

usage:
    Tableau format:
        _matrix:
            1 -<objective-function>  0
            0 <lhs>                 <rhs>
        _basis: [0, s1, s2, ...]

    i.e. for problem:
        max z = 7x + 4y,
        s.t.    2x +  y <= 20
                 x +  y <= 18
                 x      <= 8
    setup Tableau as:
        t = lp.Tableau(lp.array([
            [1, -7, -4, 0, 0, 0,  0],
            [0,  2,  1, 1, 0, 0, 20],
            [0,  1,  1, 0, 1, 0, 18],
            [0,  1,  0, 0, 0, 1,  8]
        ]), [0, 3, 4, 5])
    or
        t = lp.Tableau.make(<lhs>, <rhs>, <objective>, <basis>)

    run_primal_simplex(t: Tableau)
        requirements: rhs of all constraints must be non-negative
    run_dual_simplex(t: Tableau)
        requirements: objective function coefficients must be non-negative

    both functions edit Tableau in a way, such _basis and _matrix are set properly

    use them to solve MILP problem:
        first iteration: set up Tableau, using artificial variables
        (this way simplex method requirement is resolved)
        next iterations: solution is optimal => dual method requirement is resolved,
            so use dual simplex method may be used without any changes

'''

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum

EPS = 1e-10

Float = np.float64
NDArray = npt.NDArray[Float]

array = lambda x: np.array(x, dtype=Float)


class ConstraintSign(Enum):
    LEQ = 1  # <=
    GEQ = 2  # >=


@dataclass
class Tableau:
    _matrix: NDArray
    _basis: list

    @classmethod
    def make(cls, lhs: NDArray, rhs: NDArray, objective: NDArray, basis: list):
        rhs = np.reshape(rhs, (rhs.shape[0], 1))
        matrix = np.concatenate((lhs, rhs), axis=1)
        return cls.from_matrix(matrix, basis, objective)

    @classmethod
    def from_matrix(cls, matrix: NDArray, basis: list, func: NDArray):
        func = np.hstack((func, np.zeros(matrix.shape[1] - 1 - func.shape[0], dtype=Float)))
        matrix = np.insert(matrix, 0, array([0]), axis=1)
        objective = np.concatenate(([-1], func, [0]), dtype=Float)
        matrix = np.concatenate((np.reshape(-objective, (1, objective.shape[0])), matrix), dtype=Float)
        return cls(matrix, [0] + list(map(lambda x: x + 1, basis)))

    @property
    def variables_count(self) -> int:
        return self._matrix.shape[1] - 2

    def add_constraint(self, lhs: NDArray, rhs: Float, sign: ConstraintSign) -> None:
        if sign == ConstraintSign.GEQ:
            lhs, rhs = -lhs, -rhs
        cons = np.concatenate(([0], lhs, [1], [rhs]), dtype=Float)
        self._basis.append(lhs.shape[0] + 1)
        self._matrix = np.insert(self._matrix, -1, 0, axis=1)
        self._matrix = np.append(self._matrix, values=np.reshape(cons, (1, cons.size)), axis=0)

    def add_restriction(self, index: int, value: Float, sign: ConstraintSign) -> None:
        # add lower-bound & upper-bound restriction example
        #   solution = t.solution()[1][index]
        #   t1.add_restriction(index, np.round(solution), lp.ConstraintSign.LEQ)
        #   t2.add_restriction(index, np.round(solution + 1), lp.ConstraintSign.GEQ)
        line = self._basis.index(index + 1)
        z = np.zeros(self._matrix.shape[1] - 2, dtype=Float)
        z[index] = 1
        z -= self._matrix[line, 1:-1]
        self.add_constraint(z, value - self._matrix[line, -1], sign)

    def solution(self) -> tuple[Float, NDArray]:
        r = np.zeros(self.variables_count + 1, dtype=Float)
        r[self._basis] = self._matrix[:, -1]
        # r[0] = objective value
        # r[1:] = variables' values
        return r[0], r[1:]

    def __eq__(self, other) -> bool:
        '''
        TODO: add rtol, atol to np.isclose
        '''
        return (self._basis == other._basis and
                self._matrix.shape == other._matrix.shape and
                np.allclose(self._matrix, other._matrix))

    def make_basis_non_degenerate(self) -> None:
        a = self._matrix[0, self._basis]
        a[a == 0] = 1e-20


def normalize(t: Tableau, col: int, row: int):
    t._basis[row] = col
    r = t._matrix[row, :] / t._matrix[row, col]
    t._matrix = np.apply_along_axis(lambda x: x - r * x[col], 1, t._matrix)
    t._matrix[row, :] = r


def run_primal_simplex(t: Tableau) -> bool:
    def find_entering_variable(t: Tableau) -> int | None:
        r = t._matrix[0, :-1]  # objective coefficients
        index = int(r.argmin())
        return None if r[index] >= -EPS else index

    def find_leaving_variable(t: Tableau, v: int) -> int | None:
        col = t._matrix[:, v]
        rhs = t._matrix[:, -1]
        div = np.divide(rhs, col, where=(col * rhs >= 0) & (col != 0), out=np.full_like(col, np.inf))
        div[(rhs == 0) & (col < 0)] = np.inf
        div[0] = np.inf
        if (min := div.min()) == np.inf:
            return None  # unbounded/infeasible
        # if min < EPS and min > -EPS:
        #     return None # unbounded/infeasible
        r = np.array(t._basis, dtype=Float)
        r[np.where(div != min)] = np.inf
        return int(r.argmin())

    iter = 0

    while True:
        if (col := find_entering_variable(t)) is None:
            return True
        if (row := find_leaving_variable(t, col)) is None:
            return False
        normalize(t, col, row)
        # if (iter := iter + 1) == 10000:
        #     raise RuntimeError('too many iterations')

def run_dual_simplex(t: Tableau) -> bool:
    def find_leaving_variable(t: Tableau) -> int | None:
        r = t._matrix[1:, -1]
        index = int(r.argmin())
        return None if r[index] >= -EPS else index + 1

    def find_entering_variable(t: Tableau, v: int) -> int | None:
        row = t._matrix[v, :-1]
        obj = t._matrix[0, :-1]
        # r = np.divide(obj, row, where=row * obj > 0, out=np.full_like(row, np.inf))
        r = np.divide(obj, row, where=(row < 0) & (obj <= 0), out=np.full_like(row, np.inf))
        index = int(r.argmin())
        return None if r[index] == np.inf else index
        #
        # div = np.divide(obj, row, where=row * obj > 0, out=np.full_like(row, np.inf))
        # if (min := div.min()) == np.inf:
        #     return None  # unbounded/infeasible
        # r = np.array(t._basis, dtype=Float)
        # r[np.where(div != min)] = np.inf
        # return int(r.argmin())

    # max{f} -> min{-f}
    t._matrix[0, 1:] = -t._matrix[0, 1:]
    while True:
        if (row := find_leaving_variable(t)) is None:
            t._matrix[0, 1:] = -t._matrix[0, 1:]
            return True
        if (col := find_entering_variable(t, row)) is None:
            return False
        normalize(t, col, row)


make_solution_optimal = run_primal_simplex
make_solution_feasible = run_dual_simplex
#
# t = Tableau(array([
#     [1, -6, -9, 0, 0, 0],
#     [0, 5, -10, 1, 0, 157],
#     [0, 4,   6, 0, 1, 33]
# ]), [0, 3, 4])
#
# np.set_printoptions(suppress=True, linewidth=100)
# print(t._matrix, t._basis)
# run_primal_simplex(t)
# print(t._matrix, t._basis)
# t.add_restriction(1, Float(5), ConstraintSign.LEQ)
# print(t._matrix, t._basis)
# run_dual_simplex(t)
# print(t._matrix, t._basis)
# run_primal_simplex(t)
# print(t._matrix, t._basis)
#
