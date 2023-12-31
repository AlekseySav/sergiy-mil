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


def make_unit(x, tsks) -> Unit:
    match len(x):
        case 2:
            return Unit(x[0], [tsks[i] for i in x[1]])
        case 3:
            return Unit(x[0], [tsks[i] for i in x[1]], max_batch_size=x[2])
        case 4:
            return Unit(x[0], [tsks[i] for i in x[1]], min_batch_size=x[2], max_batch_size=x[3])


def make_arc(x) -> Arc:
    match len(x):
        case 2:
            return Arc((x[0], x[1]))
        case 3:
            return Arc((x[0], x[1]), [x[2], x[2]])
        case 4:
            return Arc((x[0], x[1]), [x[2], x[3]])


states1 = list(map(lambda x: make_state(x),
                   [
                       ["State 1", 200],
                       ["State 2", 200],
                   ]))

tasks1 = list(map(lambda x: make_task(x),
                  [
                      ["Task 1", 1]
                  ]))

units1 = list(map(lambda x: make_unit(x, tasks1),
                  [
                      ["Unit 1", [0], 100]
                  ]))

arcs1 = list(map(lambda x: make_arc(x),
                 [
                     [states1[0], tasks1[0]],
                     [tasks1[0], states1[1]]
                 ]))

PrimitiveFactory = STN(states1, tasks1, arcs1, units1)

states2 = list(map(lambda x: make_state(x),
                   [
                       ["State 1"],
                       ["State 2", 400],
                   ]))

tasks2 = list(map(lambda x: make_task(x),
                  [
                      ["Task 1", 1]
                  ]))

units2 = list(map(lambda x: make_unit(x, tasks2),
                  [
                      ["Unit 1", [0], 100],
                      ["Unit 2", [0], 200]
                  ]))

arcs2 = list(map(lambda x: make_arc(x),
                 [
                     [states2[0], tasks2[0]],
                     [tasks2[0], states2[1]]
                 ]))

TwoUnitsFactory = STN(states2, tasks2, arcs2, units2)


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

ArticleExample1 = STN(states, tasks, arcs, units)
