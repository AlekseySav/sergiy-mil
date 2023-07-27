import numpy as np
from dataclasses import dataclass

@dataclass
class Variable:
    is_int: bool
    lower_bound: float | None = None
    upper_bound: float | None = None
    value: float = float()
    displacement: float = float()


Column = np.array(float) #list[float]
Matrix = np.array(float) #list[list[float]]


def normalize(constraints: Matrix, objectives: Column, vars: list[Variable]) -> None:
    return None

def evaluate(vars: list[Variable], n: int) -> list[float]:
    return None