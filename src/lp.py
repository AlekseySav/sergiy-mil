import numpy as np
from tableau import Tableau


'''

Problem:
    Min 7x+6y, where
        2x + 4y <= 16
        3x + 2y <= 12
        x, y >= 0

Solution:
    t = Tableau(func=array([7, 6]))
    t.add_constraint(array([2, 4, 16]))
    t.add_constraint(array([3, 2, 0, 12]))
    solve_lp(t)
    print(t.solution())

Result:
    ~32
'''


'''
pending optimizations (?)

- A_b, A_b^-1, don't have to be evalueated multiple times

'''


def optimal_bfs_test(t: Tableau) -> int | None:
    c_b = t.func[t.basis]
    A_b = t.matrix[:,t.basis]
    z_j = t.matrix.T @ (np.linalg.inv(A_b).T @ c_b)
    r = t.func - z_j[:-1]
    index = r.argmin()
    return None if r[index] >= 0 else index


# t.basis[x] = column, x = row
def min_ratio_test(t: Tableau, e: int) -> int | None:
    A_e = t.matrix[:,e]
    A_b = t.matrix[:,t.basis]
    b = t.matrix[:,-1]

    p1 = np.linalg.inv(A_b) @ b
    p2 = np.linalg.inv(A_b) @ A_e
    '''
    NOTE: from p2 must be selected only ones > 0
    NOTE: special case on p1=0?
    '''
    return (p1 / p2).argmin()


# normilize basis & matrix such that m[row][col] = 1
def normilize(t: Tableau, row: int, col: int) -> None:
    t.basis[row] = col
    r = t.matrix[row,:] / t.matrix[row, col]
    t.matrix = np.apply_along_axis(lambda x: x - r * x[col], 1, t.matrix)
    t.matrix[row,:] = r


# apply simplex method
def solve_lp(t: Tableau) -> None:
    while (col := optimal_bfs_test(t)) is not None:
        row = min_ratio_test(t, col)
        normilize(t, row, col)
