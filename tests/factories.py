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

states3 = list(map(lambda x: make_state(x),
                   [
                       ["State 1"],
                       ["State 2", 400],
                   ]))

tasks3 = list(map(lambda x: make_task(x),
                  [
                      ["Task 1", 1]
                  ]))

units3 = list(map(lambda x: make_unit(x, tasks3),
                  [
                      ["Unit 1", [0], 100],
                      ["Unit 2", [0], 200]
                  ]))

arcs3 = list(map(lambda x: make_arc(x),
                 [
                     [states3[0], tasks3[0]],
                     [tasks3[0], states3[1]]
                 ]))

TwoUnitsTwoStates = STN(states3, tasks3, arcs3, units3)
