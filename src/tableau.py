import numpy as np
from dataclasses import dataclass, field


'''
Problem:
    Min/Max 7x+6y, where
        2x + 4y <= 16
        3x + 2y <= 12
        x, y >= 0

Setup:
    t = Tabletau(func=array([7, 6]))
    t.add_constraint([2, 4, 16])            # 2x + 4y <= 16
    t.add_constraint([3, 2, 0, 12])         # 3x + 2y <= 12
                            ^~~~~~~~ s1
Result after setup:
    func        7  6  0  0
    matrix      2  4  1  0  16
                3  2  0  1  12
    basis             ^  ^
    count=4
'''


def array(data=()) -> np.ndarray[np.float32]:
    return np.array(object=data, dtype=np.float32)


@dataclass
class Tableau:
    matrix: np.matrix[np.float32]
    basis: list[int]
    func: np.ndarray[np.float32]

    def __init__(self, func) -> None:
        self.basis = []
        self.func = func
        self.matrix = np.empty(shape=(0, func.size + 1), dtype=np.float32)

    @property
    def count(self) -> int:
        return self.func.size

    def solution(self) -> np.float32:
        res = 0
        for row, x in enumerate(self.basis):
            res += self.func[x] * self.matrix[row, -1]
        return res

    # NOTE: cons must not contain basic variables
    def add_constraint(self, cons: np.ndarray[np.float32]) -> None:
        self.matrix = np.append(self.matrix, values=np.reshape(cons, (1, cons.size)), axis=0)
        self.matrix = np.insert(self.matrix, -1, values=0, axis=1)
        self.matrix[-1, -2] = 1
        self.basis.append(self.count)
        self.func = np.append(self.func, 0)

    # temporary
    def print(self) -> None:
        print()
        print(f'{self.func=}')
        print(f'{self.count=}')
        print(f'{self.basis=}')
        print(f'{self.matrix}')
        print(f'{self.solution()=}')
