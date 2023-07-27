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

from src.tableau import Tableau
from src.lp.lp import solve_lp
from collections.abc import Callable


def __split_subdivision__(problem: Tableau, var_index: int) -> list[Tableau]:
    ...


def __get_solution__(problem: Tableau, constraints: list[bool]) -> list[float]:
    ...


def __check_solution__(problem: Tableau, constraints: list[bool]) -> list[bool]:
    ...


def __is_infeasible__(problem: Tableau) -> list[bool]:
    ...


def solve_milp(problem: Tableau, constraints: list[bool],
               get_tableau: Callable[[list[Tableau], list[bool]], Tableau],
               get_axis: Callable[[Tableau, list[bool]], int]) -> float:
    z_lower = float('-inf')
    z_upper = float('-inf')
    subdivisions = [problem]
    while len(subdivisions):
        problem = get_tableau(subdivisions, constraints)
        subdivisions.remove(problem)
        solve_lp(problem)
        if __is_infeasible__(problem):
            continue
        z = problem.solution()
        z_upper = max(z_upper, z)
        if z <= z_lower:
            continue
        if all(__check_solution__(problem, constraints)):
            z_lower = z
        else:
            split_index = get_axis(problem, constraints)
            if split_index is None:
                continue
            subdivisions.append(__split_subdivision__(problem, split_index))
    return z_lower
