import numpy as np
from src.tableau import Tableau


'''
interface
    - make_solution_optimal(t)  run primal simplex method
    - make_solution_feasible(t) run dual   simplex method
    - check_solution(t)         check that solution is optimal and feasible
'''


'''

Problem:
    Max 7x+6y, where
        2x + 4y <= 16
        3x + 2y <= 12
        x, y >= 0

Solution:
    t = Tableau(func=array([-7, -6]))
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

def normilize(t: Tableau, row: int, col: int) -> None:
    '''
    used by both primal and dual methods

    make variable at (row, col) basic and normilize matrix
    '''
    t.basis[row] = col
    r = t.matrix[row,:] / t.matrix[row, col]
    t.matrix = np.apply_along_axis(lambda x: x - r * x[col], 1, t.matrix)
    t.matrix[row,:] = r


'''
primal simplex method
'''

def optimal_bfs_test(t: Tableau) -> int | None:
    c_b = t.func[t.basis]
    A_b = t.matrix[:,t.basis]
    A_b_inv = np.linalg.inv(A_b)
    z_j = t.matrix.T @ (A_b_inv.T @ c_b)
    r = t.func - z_j[:-1]
    index = r.argmin()
    return None if r[index] >= 0 else index

def primal_min_ratio_test(t: Tableau, e: int) -> int | None:
    A_e = t.matrix[:,e]
    A_b = t.matrix[:,t.basis]
    A_b_inv = np.linalg.inv(A_b)
    b = t.matrix[:,-1]
    p = np.divide(A_b_inv @ b, A_b_inv @ A_e, np.zeros_like(b), where=A_e != 0)
    '''
    NOTE: case when multiple zeroes (?) when zero (?)
    '''
    index = np.where(p > 0, p, np.inf).argmin()
    return None if p[index] <= 0 else index

def make_solution_optimal(t: Tableau) -> bool:
    '''
    run primal simplex method

    returns `True` if solution is bounded

    #### requirements:
        - `A_b @ c_b` is a feasible solution
        - `A[-1]` is fotmated as `s_n = N`

    #### designations:
        - `A_b`: `t.matrix[:,t.basis]`
        - `c_b`: `t.func[t.basis]`
    '''
    while (col := optimal_bfs_test(t)) is not None:
        row = primal_min_ratio_test(t, col)
        if row is None:
            return False
        normilize(t, row, col)
    return True


'''
dual simplex method
'''

def feasible_rhs_test(t: Tableau) -> int | None:
    r = t.matrix[:,-1]
    index = r.argmin()
    return None if r[index] >= 0 else index

def dual_min_ratio_test(t: Tableau, e: int) -> int | None:
    A_e = t.matrix[e,:-1]
    p = np.divide(-t.func, A_e, np.full_like(t.func, np.Inf), where=A_e != 0)
    return p.argmin()

def make_solution_feasible(t: Tableau) -> None:
    '''
    run dual simplex method

    #### requirements:
        - `A_b @ c_b` is an optimal solution
        - `A[-1]` is fotmated as `-A * x_j + ... + s_n = -B`
        - solution is bounded (i.e. `t.func >= 0` or `optimize_solution` returned `True`)

    #### designations:
        - `A_b`: `t.matrix[:,t.basis]`
        - `c_b`: `t.func[t.basis]`
    '''
    while (row := feasible_rhs_test(t)) is not None:
        col = dual_min_ratio_test(t, row)
        normilize(t, row, col)
    return True


'''
check_solution
'''

def check_solution(t: Tableau) -> bool:
    '''
    check that given solution is optimal and feasible
    '''
    return optimal_bfs_test(t) is None and feasible_rhs_test(t) is None


'''
solve_lp: temporary function
'''
def solve_lp(t):
    return make_solution_optimal(t)



# from tableau import array

# t = Tableau.from_matrix(
#     matrix=array([
#         [ 2,  1, 1, 0, 0, 0,  600],
#         [ 1,  1, 0, 1, 0, 0,  225],
#         [ 5,  4, 0, 0, 1, 0, 1000],
#         [-1, -2, 0, 0, 0, 1, -150]
#     ]),
#     basis=[2, 3, 4, 5],
#     func=array([3, 4, 0, 0, 0, 0])
# )

# t = Tableau.from_matrix(
#     matrix=array([
#         [1.,  0.,  2.25, -0.25,  0.,  0.,  0.,  2.25],
#         [0.,  1., -1.25,  0.25,  0.,  0.,  0.,  3.75],
#         [0.,  0.,  2.25, -0.25,  1.,  0.,  0.,  2.25],
#         [0.,  0., -1.25,  0.25,  0.,  1.,  0.,  3.75],
#         [0.,  0., -2.25,  0.25,  0.,  0.,  1., -0.25]]),
#     basis=[0, 1, 4, 5, 6],
#     func=array([-5., -8.,  0.,  0.,  0.,  0.,  0.])
# )

# make_solution_feasible(t)
# print(t.solution())
