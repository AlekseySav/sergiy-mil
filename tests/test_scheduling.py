import factories
import or_tools_api
from scheduling_problem import *


def solution_or_tools(factory, d, e, filename: str | None = None) \
        -> tuple[float | None, dict[str, float] | None, str]:
    sp = SP(factory, d, e)

    if filename is not None:
        sp.draw('factory_diagrams/' + filename)

    sp._generate_problem()
    vars_values = []
    res, output = or_tools_api.solve_or_tools(sp.problem, result_values=vars_values)
    if res is None:
        return None, None, output
    solution = dict()
    for u in sp.U:
        for i in sp.I_u[u]:
            for t in range(sp.H):
                solution[f'Q[{u.name}][{i.name}][{t}]'] = vars_values[sp.Q[u][i][t].index]
    return res, solution, output


def string(solution: dict[str, float]) -> str:
    res = "Solution:\n"
    for s in solution.keys():
        res += f'{s}={solution[s] if abs(solution[s]) > 1/1000000 else 0}\n'
    res += '\n'
    return res


def test_PrimitiveFactory_trivial_1():
    """
    Factory schema like:
        state_1(input, unlimited) -> task1 (unit1) -> state_2 (output,low requirements)
    """

    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100]},
                                             {f.states[0]: [100, 0]},
                                             'PrimitiveFactory.dot')
    assert ms == -1, string(solution) + output


def test_PrimitiveFactory_trivial_2():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 0, 100]},
                                             {f.states[0]: [200, 0, 0, 0]},
                                             'PrimitiveFactory.dot')
    assert ms == -2, string(solution) + output


def test_PrimitiveFactory_trivial_3():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 0, 100]},
                                             {f.states[0]: [100, 0, 100, 0]},
                                             'PrimitiveFactory.dot')
    assert ms == -3, string(solution) + output


def test_PrimitiveFactory_impossible_1():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 100]},
                                             {f.states[0]: [100, 0, 0]},
                                             'PrimitiveFactory.dot')
    assert ms is None, string(solution) + output


def test_PrimitiveFactory_impossible_2():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 0]},
                                             {f.states[0]: [0, 100, 100]},
                                             'PrimitiveFactory.dot')
    assert ms is None, string(solution) + output


def test_PrimitiveFactory_impossible_3():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 101, 0]},
                                             {f.states[0]: [200, 0, 0]},
                                             'PrimitiveFactory.dot')
    assert ms is None, string(solution) + output


def test_PrimitiveFactory_input_overflow():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 0, 0, 100]},
                                             {f.states[0]: [100, 100, 100, 100]},
                                             'PrimitiveFactory.dot')
    assert ms == -2, string(solution) + output


def test_two_units_factory_trivial_1():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 300]},
                                             {f.states[0]: [300, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms == -1, string(solution) + output


def test_two_units_factory_trivial_2():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 0, 450]},
                                             {f.states[0]: [225, 225, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms == -2, string(solution) + output


def test_two_units_factory_output_limit():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400, 0, 400]},
                                             {f.states[0]: [800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms == -11, string(solution) + output


def test_two_units_factory_impossible_1():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 0, 0, 0, 0, 0, 0, 400, 0, 0, 400, 0, 400]},
                                             {f.states[0]: [800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms is None, string(solution) + output


def test_two_units_factory_impossible_2():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 400]},
                                             {f.states[0]: [800, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms is None, string(solution) + output


def test_two_units_factory_impossible_3():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 300, 0]},
                                             {f.states[0]: [200, 100, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms is None, string(solution) + output


def test_article_ex_factory():
    f = factories.ArticleExample1

    ms, solution, output = solution_or_tools(f,
                                             {
                                                 f.states[8]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               100],
                                                 f.states[9]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               100]
                                             },
                                             {
                                                 f.states[0]: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               0],
                                                 f.states[1]: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               0],
                                                 f.states[2]: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                               0], },
                                             'ArticleExample1.dot')
    assert ms == -9, string(solution) + output
