from copy import copy

from src.tableau import Tableau, array


class Problem:
    constraint_coeffs: list[list[float]]
    bounds: list[float]
    obj_coeffs: list[float]
    num_vars: int
    num_constraints: int

    def __init__(self, constraint_coeffs: list[list[float]], bounds: list[float], obj_coeffs: list[float]):
        self.constraint_coeffs = constraint_coeffs
        self.bounds = bounds
        self.obj_coeffs = obj_coeffs
        self.num_vars = len(constraint_coeffs[0])
        self.num_constraints = len(bounds)

    def to_dict(self) -> dict:
        """Stores the data for the problem."""
        data = {'constraint_coeffs': self.constraint_coeffs, 'bounds': self.bounds, 'obj_coeffs': self.obj_coeffs,
                'num_vars': self.num_vars, 'num_constraints': self.num_constraints}
        return data

    def to_tableau(self) -> Tableau:
        zeroes = [0 for _ in range(self.num_constraints + 1)]
        matrix = [
            copy(self.constraint_coeffs[i]) + copy(zeroes) for i in range(self.num_constraints)
        ]
        for i in range(self.num_constraints):
            matrix[i][self.num_vars + i] = 1
            matrix[i][-1] = self.bounds[i]
        basis = [self.num_vars + i for i in range(self.num_constraints)]
        return Tableau.from_matrix(
            matrix=array(matrix),
            basis=basis,
            func=array(copy(self.obj_coeffs)+zeroes[:-1])
        )
