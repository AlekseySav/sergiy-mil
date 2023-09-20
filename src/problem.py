from copy import copy

from lp import Tableau, array
from milp import solve_milp

BIG_M = 1000000


class Problem:
    def __init__(self, constraint_coeffs: list[list[float]], bounds: list[float], obj_coeffs: list[float],
                 type_constaints: list[bool] | None = None):
        self.solution = None
        self.output = None
        self.solution_value = None
        self.constraint_coeffs = constraint_coeffs
        self.bounds = bounds
        self.obj_coeffs = obj_coeffs
        self.num_vars = len(constraint_coeffs[0])
        self.num_constraints = len(bounds)
        self.used_big_m = False
        if type_constaints is None:
            self.type_constraints = [False for _ in range(self.num_vars)]
        else:
            self.type_constraints = type_constaints

    def to_dict(self) -> dict:
        data = {'constraint_coeffs': self.constraint_coeffs, 'bounds': self.bounds, 'obj_coeffs': self.obj_coeffs,
                'num_vars': self.num_vars, 'num_constraints': self.num_constraints}
        return data

    def fix_ineq_with_righ_part_negative(self):
        negative_bound = 0
        for i in self.bounds:
            if i < 0:
                negative_bound = min(negative_bound, i)
        if negative_bound < 0:
            self.obj_coeffs.append(BIG_M)
            for i in range(self.num_constraints):
                if self.bounds[i] < 0:
                    self.constraint_coeffs[i].append(1)
                    self.bounds[i] += abs(negative_bound)
                else:
                    self.constraint_coeffs[i].append(0)
            self.constraint_coeffs.append([0 for _ in range(self.num_vars)])
            self.constraint_coeffs[-1].append(1)
            self.bounds.append(abs(negative_bound))
            self.num_constraints += 1
            self.num_vars += 1
            self.type_constraints.append(False)
            self.used_big_m = True

    def to_tableau(self) -> Tableau:
        self.fix_ineq_with_righ_part_negative()
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
            func=array(copy(self.obj_coeffs) + zeroes[:-1])
        )

    def solve(self, gt, ga):
        self.solution = []
        self.solution_value = None
        self.output = ""
        self.solution_value, self.output = solve_milp(self.to_tableau(), self.type_constraints, gt, ga,
                                                      solution=self.solution)
        if self.used_big_m and self.solution_value is not None:
            self.solution_value -= BIG_M * self.solution[-1]
            self.solution = self.solution[:-1]

    def __str__(self) -> str:
        return (f'\n\nProblem: \nconstraints coeffs: {self.constraint_coeffs}\nbounds: {self.bounds}\nobj_coeffs: '
                f'{self.obj_coeffs}')
