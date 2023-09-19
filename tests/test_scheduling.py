import numpy as np

import or_tools_api
from milp import solve_milp
from or_tools_api import solve_or_tools
from scheduling_problem import *


def make_state(x) -> State:
    if len(x) == 1:
        return State(name=x[0])
    if len(x) == 2:
        return State(name=x[0], max_stock=x[1])
    if len(x) == 3:
        return State(name=x[0], min_stock=x[1], max_stock=x[2])
    if len(x) == 4:
        return State(name=x[0], min_stock=x[1], max_stock=x[2], ns=x[3])


def make_task(x) -> Task:
    match len(x):
        case 1:
            return Task(name=x[0])
        case 2:
            return Task(name=x[0], batch_processing_time=x[1])


def make_unit(x, tasks) -> Unit:
    match len(x):
        case 2:
            return Unit(x[0], [tasks[i] for i in x[1]])
        case 3:
            return Unit(x[0], [tasks[i] for i in x[1]], max_batch_size=x[2])
        case 4:
            return Unit(x[0], [tasks[i] for i in x[1]], min_batch_size=x[2], max_batch_size=x[3])


def make_arc(x) -> Arc:
    match len(x):
        case 2:
            return Arc([x[0], x[1]])
        case 3:
            return Arc([x[0], x[1]], [x[2], x[2]])
        case 4:
            return Arc([x[0], x[1]], [x[2], x[3]])


def test():
    states = [State()]
    tasks = [Task()]
    units = [Unit(tasks=tasks)]
    arcs = [Arc([states[0], tasks[0]])]
    stn = STN(states, tasks, arcs, units)
    stn.draw('0.dot')
    sp = SP(states, tasks, arcs, units, {states[0]: [1, 1, 1, 1, 1]},
            {states[0]: [1000, 1000, 1000, 1000, 1000]})
    sp.generate_dictionaries()
    sp.generate_variables()
    sp.generate_constraints()
    sp.generate_problem()
    print(solve_or_tools(sp.problem))

    states = list(map(lambda x: make_state(x),
                      [
                          ["State 1"],
                          ["State 2"],
                          ["State 3"],
                          ["State 4", 100],
                          ["State 5", 200],
                          ["State 6", 0, INF, True],
                          ["State 7", 150],
                          ["State 8", 0, INF, True],
                          ["State 9"],
                          ["State 10"],
                      ]))
    tasks = list(map(lambda x: make_task(x),
                     [
                         ["Task 1", 1],
                         ["Task 2", 2],
                         ["Task 3", 2],
                         ["Task 4", 1],
                         ["Task 5", 1],
                         ["Task 6", 2],
                     ]))
    units = list(map(lambda x: make_unit(x, tasks),
                     [
                         ["Unit 1", [0], 100],
                         ["Unit 2", [1, 2, 3], 50],
                         ["Unit 3", [1, 2, 3], 90],
                         ["Unit 4", [4, 5], 200]
                     ]))
    arcs = list(map(lambda x: make_arc(x),
                    [
                        [states[0], tasks[0]],
                        [tasks[0], states[3]],
                        [states[3], tasks[1], 0.4],
                        [states[1], tasks[2], 0.5],
                        [states[2], tasks[2], 0.5],
                        [states[2], tasks[3], 0.2],
                        [tasks[1], states[8], 0.4],
                        [tasks[1], states[6], 0.6],
                        [tasks[2], states[4]],
                        [tasks[3], states[5]],
                        [states[6], tasks[3], 0.8],
                        [states[4], tasks[1], 0.6],
                        [states[5], tasks[4]],
                        [tasks[4], states[7], 0.1],
                        [states[7], tasks[5]],
                        [tasks[4], states[9], 0.9],
                        [tasks[5], states[6]]
                    ]))
    sp = SP(states, tasks, arcs, units, {states[8]: [0, 1, 1, 1, 1],
                                         states[9]: [0, 1, 1, 1, 1]},
            {states[0]: [10, 10, 10, 10, 10],
             states[1]: [10, 10, 10, 10, 10],
             states[2]: [10, 10, 10, 10, 10],
             })
    sp.draw('article_example.dot')
    sp.generate_dictionaries()
    sp.generate_variables()
    sp.generate_constraints()
    sp.generate_problem()
    print(solve_or_tools(sp.problem))


def test_primitive_factory():
    """
    Factory schema like:
        state_1(input, unlimited) -> task1 (unit1) -> state_2 (output,low requirements)
    """

    states = list(map(lambda x: make_state(x),
                      [
                          ["State 1", 200],
                          ["State 2", 200],
                      ]))

    tasks = list(map(lambda x: make_task(x),
                     [
                         ["Task 1", 1]
                     ]))

    units = list(map(lambda x: make_unit(x, tasks),
                     [
                         ["Unit 1", [0], 100]
                     ]))

    arcs = list(map(lambda x: make_arc(x),
                    [
                        [states[0], tasks[0]],
                        [tasks[0], states[1]]
                    ]))
    sp = SP(states, tasks, arcs, units,
            {states[1]: [0, 0, 0, 100]},
            {states[0]: [100, 100, 100, 100]}) # IT IS OKAY that answer is 2

    sp.draw('primitive_factory.dot')
    sp.generate_dictionaries()
    sp.generate_variables()
    sp.generate_constraints()
    sp.generate_problem()
    vars_values = []
    res, output = or_tools_api.solve_or_tools(sp.problem, vars_values)
    # m_res, m_output = solve_milp(sp.problem.to_tableau(), sp.problem.type_constraints)
    print(res)
    print(output)
    print(sp.problem)
    if res is None:
        return
    for u in sp.U:
        for i in sp.I_u[u]:
            for t in range(sp.H):
                print(f'x[{u.name}][{i.name}][{t}]={vars_values[sp.x[u][i][t].index]}')
    for u in sp.U:
        for i in sp.I_u[u]:
            for t in range(sp.H):
                print(f'Q[{u.name}][{i.name}][{t}]={vars_values[sp.Q[u][i][t].index]}')


def helper_for_factory(states, tasks, units, arcs, d, e):
    ...
