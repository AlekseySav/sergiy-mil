import numpy as np

import factories
import or_tools_api
from milp import solve_milp
from or_tools_api import solve_or_tools
from scheduling_problem import *


# def test():
#     states = [State()]
#     tasks = [Task()]
#     units = [Unit(tasks=tasks)]
#     arcs = [Arc([states[0], tasks[0]])]
#     stn = STN(states, tasks, arcs, units)
#     stn.draw('0.dot')
#     sp = SP(states, tasks, arcs, units, {states[0]: [1, 1, 1, 1, 1]},
#             {states[0]: [1000, 1000, 1000, 1000, 1000]})
#     sp.generate_problem()
#     print(solve_or_tools(sp.problem))
#
#     states = list(map(lambda x: make_state(x),
#                       [
#                           ["State 1"],
#                           ["State 2"],
#                           ["State 3"],
#                           ["State 4", 100],
#                           ["State 5", 200],
#                           ["State 6", 0, INF, True],
#                           ["State 7", 150],
#                           ["State 8", 0, INF, True],
#                           ["State 9"],
#                           ["State 10"],
#                       ]))
#     tasks = list(map(lambda x: make_task(x),
#                      [
#                          ["Task 1", 1],
#                          ["Task 2", 2],
#                          ["Task 3", 2],
#                          ["Task 4", 1],
#                          ["Task 5", 1],
#                          ["Task 6", 2],
#                      ]))
#     units = list(map(lambda x: make_unit(x, tasks),
#                      [
#                          ["Unit 1", [0], 100],
#                          ["Unit 2", [1, 2, 3], 50],
#                          ["Unit 3", [1, 2, 3], 90],
#                          ["Unit 4", [4, 5], 200]
#                      ]))
#     arcs = list(map(lambda x: make_arc(x),
#                     [
#                         [states[0], tasks[0]],
#                         [tasks[0], states[3]],
#                         [states[3], tasks[1], 0.4],
#                         [states[1], tasks[2], 0.5],
#                         [states[2], tasks[2], 0.5],
#                         [states[2], tasks[3], 0.2],
#                         [tasks[1], states[8], 0.4],
#                         [tasks[1], states[6], 0.6],
#                         [tasks[2], states[4]],
#                         [tasks[3], states[5]],
#                         [states[6], tasks[3], 0.8],
#                         [states[4], tasks[1], 0.6],
#                         [states[5], tasks[4]],
#                         [tasks[4], states[7], 0.1],
#                         [states[7], tasks[5]],
#                         [tasks[4], states[9], 0.9],
#                         [tasks[5], states[6]]
#                     ]))
#     sp = SP(states, tasks, arcs, units, {states[8]: [0, 0, 0, 0, 0],
#                                          states[9]: [0, 0, 0, 0, 0]},
#             {states[0]: [10, 10, 10, 10, 10],
#              states[1]: [10, 10, 10, 10, 10],
#              states[2]: [10, 10, 10, 10, 10],
#              })
#     sp.draw('article_example.dot')
#     sp.generate_problem()
#     print(solve_or_tools(sp.problem))


def solution_or_tools(factory, d, e, filename: str | None = None) \
        -> tuple[float | None, dict[str, float] | None, str]:
    sp = SP(factory, d, e)

    if filename is not None:
        sp.draw(filename)

    sp.generate_problem()
    vars_values = []
    res, output = or_tools_api.solve_or_tools(sp.problem, result_values=vars_values)
    if res is None:
        return None, None, output
    solution = dict()
    for u in sp.U:
        for i in sp.I_u[u]:
            for t in range(sp.H):
                solution[f'x[{u.name}][{i.name}][{t}]'] = vars_values[sp.x[u][i][t].index]
    for u in sp.U:
        for i in sp.I_u[u]:
            for t in range(sp.H):
                solution[f'Q[{u.name}][{i.name}][{t}]'] = vars_values[sp.Q[u][i][t].index]
    return res, solution, output


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
    assert ms == -1, str(solution) + output


def test_PrimitiveFactory_trivial_2():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 0, 100]},
                                             {f.states[0]: [200, 0, 0, 0]},
                                             'PrimitiveFactory.dot')
    assert ms == -2, str(solution) + output


def test_PrimitiveFactory_trivial_3():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 0, 100]},
                                             {f.states[0]: [100, 0, 100, 0]},
                                             'PrimitiveFactory.dot')
    assert ms == -3, str(solution) + output


def test_PrimitiveFactory_impossible_1():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 100]},
                                             {f.states[0]: [100, 0, 0]},
                                             'PrimitiveFactory.dot')
    assert ms is None, str(solution) + output


def test_PrimitiveFactory_impossible_2():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 100, 0]},
                                             {f.states[0]: [0, 100, 100]},
                                             'PrimitiveFactory.dot')
    assert ms is None, str(solution) + output


def test_PrimitiveFactory_impossible_3():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 101, 0]},
                                             {f.states[0]: [200, 0, 0]},
                                             'PrimitiveFactory.dot')
    assert ms is None, str(solution) + output


def test_PrimitiveFactory_input_overflow():
    f = factories.PrimitiveFactory

    ms, solution, output = solution_or_tools(factories.PrimitiveFactory,
                                             {f.states[1]: [0, 0, 0, 100]},
                                             {f.states[0]: [100, 100, 100, 100]},
                                             'PrimitiveFactory.dot')
    assert ms == -2, str(solution) + output


def test_two_units_factory_trivial_1():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 300]},
                                             {f.states[0]: [300, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms == -1, str(solution) + output


def test_two_units_factory_trivial_2():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 0, 450]},
                                             {f.states[0]: [225, 225, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms == -2, str(solution) + output


def test_two_units_factory_output_limit():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400, 0, 400]},
                                             {f.states[0]: [800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms == -11, str(solution) + output


def test_two_units_factory_impossible_1():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 0, 0, 0, 0, 0, 0, 400, 0, 0, 400, 0, 400]},
                                             {f.states[0]: [800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms is None, str(solution) + output


def test_two_units_factory_impossible_2():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 400]},
                                             {f.states[0]: [800, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms is None, str(solution) + output


def test_two_units_factory_impossible_3():
    f = factories.TwoUnitsFactory

    ms, solution, output = solution_or_tools(f,
                                             {f.states[1]: [0, 300, 0]},
                                             {f.states[0]: [200, 100, 0]},
                                             'TwoUnitsFactory.dot')
    assert ms is None, str(solution) + output
