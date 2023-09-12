import math

from src.problem import Problem
from tests.value_gen import ValueGenerator


class ProblemGenerator:
    generators: dict[str:ValueGenerator]
    vars: int
    cons: int

    def __init__(self, modes: dict[str:str],
                 values: dict[str:list[float]],
                 variables_count: int,
                 constraints_count: int):
        self.vars = variables_count
        self.cons = constraints_count
        self.generators = {}
        for key in modes.keys():
            if key not in values.keys():
                self.generators[key] = ValueGenerator(mode=modes[key])
            if len(values[key]) == 1:
                self.generators[key] = ValueGenerator(mode=modes[key], E=values[key][0])
            if len(values[key]) == 2:
                self.generators[key] = ValueGenerator(mode=modes[key], E=values[key][0], D=values[key][1])

    def value(self) -> Problem:
        constraints_coeffs = [
            [math.floor(self.generators['constraints_coeffs'].value()) for _ in range(self.vars)]
            for _ in range(self.cons)
        ]
        bounds = [
            math.floor(self.generators['bounds'].value()) for _ in range(self.cons)
        ]
        obj_coeffs = [
            math.floor(self.generators['obj_coeffs'].value()) for _ in range(self.vars)
        ]
        return Problem(constraints_coeffs, bounds, obj_coeffs)
