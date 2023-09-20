from copy import copy

from lp import Tableau, array


class Problem:
    constraint_coeffs: list[list[float]]
    bounds: list[float]
    obj_coeffs: list[float]
    num_vars: int
    num_constraints: int
    type_constraints: list[bool]

    def __init__(self, constraint_coeffs: list[list[float]], bounds: list[float], obj_coeffs: list[float], type_constaints: list[bool] | None = None):
        self.constraint_coeffs = constraint_coeffs
        self.bounds = bounds
        self.obj_coeffs = obj_coeffs
        self.num_vars = len(constraint_coeffs[0])
        self.num_constraints = len(bounds)
        if type_constaints is None:
            self.type_constraints = [False for _ in range(self.num_vars)]
        else:
            self.type_constraints = type_constaints

    def to_dict(self) -> dict:
        data = {'constraint_coeffs': self.constraint_coeffs, 'bounds': self.bounds, 'obj_coeffs': self.obj_coeffs,
                'num_vars': self.num_vars, 'num_constraints': self.num_constraints}
        return data

    def to_tableau(self) -> Tableau:
        zeroes = [0 for _ in range(self.num_constraints + 1)]
        matrix = [
            self.constraint_coeffs[i] + zeroes for i in range(self.num_constraints)
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

    def __str__(self) -> str:
        return (f'\n\nProblem: \nconstraints coeffs: {self.constraint_coeffs}\nbounds: {self.bounds}\nobj_coeffs: '
                f'{self.obj_coeffs}')
