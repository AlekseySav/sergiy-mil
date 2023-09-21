import tests.factories
from heuristic import ga_simple, gt_max
from milp import solve_milp
from scheduling_problem import SP


def solve():
    f = tests.factories.PrimitiveFactory
    sp = SP(f,
            {f.states[1]: [0, 100, 0, 100]},
            {f.states[0]: [100, 0, 100, 0]}
            )

    sp.solve(gt_max, ga_simple)
    print(abs(sp.solution_value))
    print(sp.solution)
    print(sp.q_dict_str())


if __name__ == '__main__':
    solve()
