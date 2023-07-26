import numpy as np
from dataclasses import dataclass

@dataclass
class Tableau:
    matrix: np.matrix[np.float32]
    basis: list[int]
    func: np.ndarray[np.float32]

    @property
    def count(self) -> int:
        return self.func.size

    def solution(self) -> np.float32:
        ...
        # res = 0
        # for row, x in enumerate(self.basis):
            # res += self.func[x] * self.matrix[row, -1]
        # return res

    # NOTE: cons must not contain basic variables
    def add_constraint(self, cons: np.ndarray[np.float32]) -> None:
        ...
        # self.matrix = np.append(self.matrix, values=cons)
        # self.matrix = np.insert(self.matrix, self.count, values=0, axis=1)
        # self.matrix[self.count, -2] = 1
        # self.basis.append(self.count)
        # self.func = np.append(self.func, 0)

    def print(self) -> None:
        ...
