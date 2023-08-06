import numpy as np
from dataclasses import dataclass


'''
Problem:
    Min/Max 7x+6y, where
        2x + 4y <= 16
        3x + 2y <= 12
        x, y >= 0

Setup:
    t = Tableau(func=array([7, 6]))
    t.add_constraint(array([2, 4, 16]))            # 2x + 4y <= 16
    t.add_constraint(array([3, 2, 0, 12]))         # 3x + 2y <= 12
                            ^~~~~~~~ s1
Result after setup:
    func        7  6  0  0
    matrix      2  4  1  0  16
                3  2  0  1  12
    basis             ^  ^
    count=4
'''


def array(data=()) -> np.ndarray:
    return np.array(object=data, dtype=np.float64)


@dataclass
class Tableau:
    '''
    TODO:
        - remove __init__
        - remove from_matrix
        - remove print()
        - add nice formatting
    '''

    matrix: np.ndarray
    basis: list[int]
    func: np.ndarray

    def __init__(self, func) -> None:
        self.basis = []
        self.func = func
        self.matrix = np.empty(shape=(0, func.size + 1), dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix, basis, func):
        t = cls(func)
        t.basis = basis
        t.matrix = matrix
        return t

    @property
    def variables_count(self) -> int:
        return self.func.size

    def solution(self) -> np.float64:
        res = 0
        for row, x in enumerate(self.basis):
            res += self.func[x] * self.matrix[row, -1]
        return res

    # NOTE: cons must not contain basic variables
    def add_constraint(self, cons: np.ndarray, leq: bool) -> None:
        self.matrix = np.append(self.matrix, values=np.reshape(cons, (1, cons.size)), axis=0)
        self.matrix = np.insert(self.matrix, -1, values=0, axis=1)
        self.matrix[-1, -2] = 1 if leq else -1
        self.basis.append(self.variables_count)
        self.func = np.append(self.func, 0)

    def __eq__(self, other):
        '''
        TODO: add rtol, atol to np.isclose
        '''
        return (self.basis == other.basis and
                np.allclose(self.func, other.func) and
                self.matrix.shape == other.matrix.shape and
                np.allclose(self.matrix, other.matrix))

    # temporary
    def print(self) -> None:
        print()
        print(f'{self.func=}')
        print(f'{self.variables_count=}')
        print(f'{self.basis=}')
        print(f'{self.matrix}')
        print(f'{self.solution()=}')
