# from src.lp.lp import Variable
# from src.lp.lp import Column
# from src.lp.lp import Matrix
#
#
# class Solver:
#     """
#     class Solver implements simplified version of Google OR-tools API for MILP (but works only with string constraints)
#     """
#
#     vars: dict[str, Variable]
#     vars_column_id: dict[str, int]
#     constraight_coeffs: Matrix
#     bounds: Column
#     obj_coeffs: Column
#
#     def __init__(self, type_: str):
#         self.type = type_
#         self.vars = {}
#         self.vars_column_id = {}
#         self.constraight_coeffs = []
#         self.bounds = []
#         self.obj_coeffs = []
#
#     def IntVar(self, lower_bound: float | None, upper_bound: float | None, name: str):
#         self.vars[name] = Variable(True, lower_bound=lower_bound, upper_bound=upper_bound)
#         self.vars_column_id[name] = len(self.vars) - 1
#
#     def BoolVar(self, name: str):
#         self.vars[name] = Variable(True, lower_bound=float(0), upper_bound=float(1))
#         self.vars_column_id[name] = len(self.vars) - 1
#
#     def NumVar(self, lower_bound: float | None, upper_bound: float | None, name: str):
#         self.vars[name] = Variable(False, lower_bound=lower_bound, upper_bound=upper_bound)
#         self.vars_column_id[name] = len(self.vars) - 1
#
#     @staticmethod
#     def infinity():
#         return None
#
#     def Add(self, constraint: str, constr=True):
#         constraint_parts = constraint.split()
#         new_constraint = [float(0) for _ in range(len(self.vars))]
#         mult = 1
#         if constr:
#             if constraint_parts[-2] == ">=":
#                 mult = -1
#             self.bounds.append(mult * float(constraint_parts[-1]))
#             constraint_parts = constraint_parts[:-2]
#
#         for i in range(len(constraint_parts)):
#             substr = constraint_parts[i]
#             if substr in self.vars_column_id.keys():
#                 if i == 0 or constraint_parts[i - 1] == "+":
#                     new_constraint[self.vars_column_id[substr]] = mult * float(1)
#                 elif constraint_parts[i - 1] == "-":
#                     new_constraint[self.vars_column_id[substr]] = mult * float(-1)
#                 elif constraint_parts[i - 1] == "*":
#                     new_constraint[self.vars_column_id[substr]] = mult * float(constraint_parts[i - 2])
#
#         if constr:
#             self.constraight_coeffs.append(new_constraint)
#         else:
#             self.obj_coeffs = new_constraint
#
#     def Maximize(self, constraint: str):
#         self.Add(constraint, constr=False)
#
#     def Minimize(self, constraint: str):
#         self.Maximize(constraint)
#         for i in range(len(self.obj_coeffs)):
#             self.obj_coeffs[i] *= -1
#
#     def Solve(self):
#         return ... #status
#
#     @classmethod
#     def CreateSolver(cls, name="MILP"):
#         return cls(name)
#
#     def __print__(self):
#         print(self.vars)
#         print(self.vars_column_id)
#         print(self.obj_coeffs)
#         print(self.bounds)
#         print(self.constraight_coeffs)
#
#
# def test():
#     solver = Solver.CreateSolver('SAT')
#     if not solver:
#         return
#
#     infinity = solver.infinity()
#     # x and y are integer non-negative variables.
#     x = solver.IntVar(0.0, infinity, 'x')
#     y = solver.IntVar(0.0, infinity, 'y')
#
#     # x + 7 * y <= 17.5.
#     solver.Add("x + 7 * y >= 17.5")
#
#     # x <= 3.5.
#     solver.Add("x <= 3.5")
#
#     # Maximize x + 10 * y.
#     solver.Maximize("x + 10 * y")
#
#     solver.__print__()
import math
from copy import copy

import numpy as np

from src.tableau import Tableau
from src.lp.lp import solve_lp
from src.milp.heuristic import func_get_tableau, func_get_axis


def __is_int__(x, delta=1 / 1000000):
    return abs(x - round(x)) < delta


def __split_subdivision__(problem: Tableau, var_index: int) -> list[Tableau]:
    p1 = copy(problem)
    p2 = copy(problem)
    n = problem.count
    constraint1 = np.zeros(n + 1)
    constraint2 = np.zeros(n + 1)
    constraint1[var_index] = 1   # x <= [value]
    constraint2[var_index] = -1  # x >= [value] + 1 <=> -x <= -[value] - 1
    row_index = -1
    for row, var in enumerate(problem.basis):
        if var == var_index:
            row_index = row
    assert row_index != -1
    value = math.floor(problem.matrix[row_index, -1])
    constraint1[-1] = value
    constraint2[-1] = -value - 1
    p1.add_constraint(constraint1)
    p2.add_constraint(constraint2)
    return [p1, p2]


def __get_solution__(problem: Tableau) -> list[float]:
    # first n columns in Tableau's matrix are correspondant to initial variables
    n = problem.count
    solution = [0.0 for _ in range(n)]
    for row, x in enumerate(problem.basis):
        if x < n:
            solution[x] = problem.matrix[row, -1]
    return solution


def __check_solution__(problem: Tableau, constraints: list[bool]) -> list[bool]:
    solution = __get_solution__(problem)
    return [not constraint or __is_int__(value)
            for constraint, value in zip(constraints, solution)]


def solve_milp(problem: Tableau, constraints: list[bool],
               get_tableau: func_get_tableau,
               get_axis: func_get_axis) -> float:
    z_lower = float('-inf')
    z_upper = float('-inf')
    subdivisions = [problem]
    while len(subdivisions):
        problem = get_tableau(subdivisions, constraints)
        subdivisions.remove(problem)
        if not solve_lp(problem):  # TODO : here it may die
            # the problem is infeasible
            continue
        z = problem.solution()
        z_upper = max(z_upper, z)
        if z <= z_lower:
            continue
        in_constraints = __check_solution__(problem, constraints)
        if all(in_constraints):
            z_lower = z
        else:
            split_index = get_axis(problem, in_constraints)
            if split_index is None:  # TODO : here it may die
                continue
            subdivisions.append(__split_subdivision__(problem, split_index))
    return z_lower
