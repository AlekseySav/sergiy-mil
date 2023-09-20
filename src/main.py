import tests.factories
from heuristic import gt_min, ga_simple, gt_max
from milp import solve_milp
from problem import Problem
from scheduling_problem import SP


def solve():
    factory = tests.factories.ArticleExample1
    sp = SP(factory, {
        factory.states[8]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
        factory.states[9]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    },
            {
                factory.states[0]: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                factory.states[1]: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                factory.states[2]: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            })
    sp.generate_problem()
    t = sp.problem.to_tableau()
    res, _ = solve_milp(t, sp.problem.type_constraints, gt_max, ga_simple)
    print(res)


if __name__ == '__main__':
    solve()
