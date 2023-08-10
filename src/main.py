from src.heuristic import gt_min, ga_simple
from src.milp import solve_milp
from src.problem import Problem


def solve():
    p = Problem(
        [
            [5, 7, 9, 2, 1],
            [18, 4, -9, 10, 12],
            [4, 7, 3, 8, 5],
            [5, 13, 16, 3, -7],
        ],
        [250, 285, 211, 315],
        [-7, -8, -2, -9, -6]
    )
    t = p.to_tableau()
    constraints = [True, True, True, True, True]
    my_res = solve_milp(t, constraints=constraints, get_tableau=gt_min, get_axis=ga_simple)
    print(my_res)


if __name__ == '__main__':
    solve()
