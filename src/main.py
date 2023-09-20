import tests.factories
from heuristic import ga_simple, gt_max
from milp import solve_milp
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

    # f = tests.factories.PrimitiveFactory
    # sp = SP(f,
    #         {f.states[1]: [0, 100]},
    #         {f.states[0]: [100, 0]}
    #         )

    sp.solve(gt_max, ga_simple)
    print(sp.solution_value)
    print(sp.solution)
    print(sp.q_dict_str())


if __name__ == '__main__':
    solve()
