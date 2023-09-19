from enum import Enum

import pydot

from problem import Problem
from tabular_lp import NDArray, array

""" 
The problem is represented in this article: https://doi.org/10.1080/002075400189004

The problem is to minimize time needed to produce required amounts of different chemicals (products).
The products can be produced on Units, using different processes (for example: heating, mixing) and other chemicals.

The input schema consists of three types of entities:
 - States - represent the chemicals - inputs and outputs from Units and their Tasks;
 - Tasks - represent different chemicals processes, taking materials from states as inputs
 and producing product's states as outputs;
 - Arcs - connect states and tasks in which they can be used as an input and connect tasks with their outputs.

The diagram looks like a graph whose nodes consist of states and tasks connected by edges represented by arcs.
"""


class State:
    name: str
    initial_stock: float
    min_stock: float
    max_stock: float
    ns: bool  # stays for non-storable

    def __init__(self, name="state0", initial_stock=0, min_stock=0, max_stock=float('+inf'), ns=False):
        self.name = name
        self.initial_stock = initial_stock
        self.min_stock = min_stock
        self.max_stock = max_stock
        self.ns = ns


class Task:
    name: str
    batch_processing_time: int

    def __init__(self, name="task0", batch_processing_time=0):
        self.name = name
        self.batch_processing_time = batch_processing_time


class Arc:
    from_to: tuple[State, Task] | tuple[Task, State]
    fraction: list[float, float]

    def __init__(self, from_to: tuple[State, Task] | tuple[Task, State], fraction=[1.0, 1.0]):
        self.from_to = from_to
        self.fraction = fraction


class Unit:
    name: str
    tasks: list[Task]
    min_batch_size: float
    max_batch_size: float

    def __init__(self, name="unit0", tasks=[], min_batch_size=0, max_batch_size=float('+inf')):
        self.name = name
        self.tasks = tasks
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size


class Variable:
    index: int

    def __init__(self, i: int):
        self.index = i


class Operator(Enum):
    LEQ = 1  # <=
    GEQ = 2  # >=
    EQ = 3  # ==


class Constraint:
    left: dict[Variable, float]
    right: float
    operator: Operator

    def __init__(self, l: dict[Variable, float], r: float = 0, operator: Operator = Operator.EQ):
        self.left = l
        self.right = r
        self.operator = operator

    def to_arrays(self, variables_count: int) -> list[NDArray]:
        if self.operator == Operator.EQ:
            self.operator = Operator.LEQ
            res = [self.to_arrays(variables_count)[0]]
            self.operator = Operator.GEQ
            res.append(self.to_arrays(variables_count)[0])
            return res
        res = array([0 for _ in range(variables_count + 1)])
        for key in self.left.keys():
            res[key.index] = self.left[key]
        res[-1] = self.right
        if self.operator == Operator.GEQ:
            res = -res
        return [res]


class STN:
    """
    State-Task-Network
    """
    states: list[State]
    tasks: list[Task]
    arcs: list[Arc]
    units: list[Unit]

    def __init__(self, states: list[State] = [], tasks: list[Task] = [], arcs: list[Arc] = [], units: list[Unit] = []):
        self.states = states
        self.tasks = tasks
        self.arcs = arcs
        self.units = units

    def draw(self, filename: str):
        tasks_to_units = {}
        tasks_to_sg = {}
        for u in self.units:
            for t in u.tasks:
                if t not in tasks_to_units.keys():
                    tasks_to_units[t] = [u]
                else:
                    tasks_to_units[t].append(u)

        graph = pydot.Dot("STN-graph", graph_type="digraph", bgcolor="white", rankdir="TB", concentrate="true")
        subgraphs = {}
        for t in self.tasks:
            sg_name = "cluster_Units: "
            for u in tasks_to_units[t]:
                sg_name += u.name + ", "
            tasks_to_sg[t] = sg_name
            s = pydot.Subgraph(sg_name, shape="box", style="dashed", label=sg_name[8:-2])
            subgraphs[sg_name] = s

        for t in self.tasks:
            task_node = pydot.Node(t.name, shape="box",
                                   label=f"{t.name}\n({t.batch_processing_time},{t.min_batch_size},{t.max_batch_size})")
            subgraphs[tasks_to_sg[t]].add_node(task_node)
        for s in subgraphs.values():
            graph.add_subgraph(s)
        for s in self.states:
            graph.add_node(pydot.Node(s.name, shape="diamond",
                                      label=f"{s.name}\n({s.initial_stock},{s.min_stock},{s.max_stock})",
                                      color=("red" if s.ns else "black")))
        for a in self.arcs:
            graph.add_edge(pydot.Edge(a.from_to[0].name, a.from_to[1].name,
                                      label=(f"{a.fraction[0]}/{a.fraction[1]}" if a.fraction[0] != a.fraction[1]
                                             else f"{a.fraction[0] if a.fraction[0] != 1 else ''}")))
        return graph.write(filename)


class SP(STN):
    U_e = None
    U_i = None
    U = None
    H = None
    J_ns = None
    J_s = None
    I_in = None
    I_u = None
    d = None
    e = None
    tau = None
    alpha_in = None
    alpha_out = None
    variables = None

    def __init__(self, states: list[State] = [], tasks: list[Task] = [], arcs: list[Arc] = [], units: list[Unit] = [],
                 d: dict[State, list[float]] = {}, e: dict[State, list[float]] = {}):
        super().__init__(states, tasks, arcs, units)
        self.constraints = None
        self.x = None
        self.p = None
        self.Q = None
        self.MS = None
        self.d = d
        self.e = e
        for x in d.values():
            self.H = len(x)
            break

    def generate_dictionaries(self):
        self.I_u: dict[Unit, set[Task]] = {}  # tasks that can be performed by unit u
        for u in self.units:
            s = set()
            for t in u.tasks:
                s.add(t)
            self.I_u[u] = s

        self.I_in: dict[State, set[Task]] = {}  # tasks that use product [State] j as input
        for a in self.arcs:
            if isinstance(a.from_to[0], State):
                if a.from_to[0] not in self.I_in.keys():
                    self.I_in[a.from_to[0]] = {a.from_to[1]}  # TODO : check
                else:
                    self.I_in[a.from_to[0]].add(a.from_to[1])

        self.I_out: dict[State, set[Task]] = {}  # tasks that produce product j
        for a in self.arcs:
            if isinstance(a.from_to[0], Task):
                if a.from_to[1] not in self.I_in.keys():
                    self.I_in[a.from_to[1]] = {a.from_to[0]}  # TODO : check
                else:
                    self.I_in[a.from_to[1]].add(a.from_to[0])

        self.J_s: set[State] = set()  # storable products
        self.J_ns: set[State] = set()  # non-storable products
        for j in self.states:
            if j.ns:
                self.J_ns.add(j)
            else:
                self.J_s.add(j)

        self.U: set[Unit] = set()  # production units
        for u in self.units:
            self.U.add(u)

        self.U_i: dict[Task, set[Unit]] = {}  # units capable of performing task i
        for u in self.units:
            for t in u.tasks:
                if t not in self.U_i.keys():
                    self.U_i[t] = {u}
                else:
                    self.U_i[t].add(u)

        J_e: set[State] = set()  # products with external demand (sum of coefficients on outgoing arcs < 1)
        for s in self.d.keys():
            J_e.add(s)

        self.U_e: set[Unit] = set()  # units producing products with external demand
        for a in self.arcs:
            if isinstance(a.from_to[0], Task):
                if a.from_to[1] in J_e:
                    for u in self.U_i[a.from_to[0]]:
                        self.U_e.add(u)

        self.alpha_in: dict[Task, dict[State, float]] = {}
        self.alpha_out: dict[
            Task, dict[State, float]] = {}  # fixed proportion of input and output of product j in task i, respectively
        for a in self.arcs:
            f = 1
            s = self.alpha_in
            if isinstance(a.from_to[0], Task):
                f = 0
                s = self.alpha_out
            if a.from_to[f] not in s.keys():
                s[a.from_to[f]] = {a.from_to[1 - f]: a.fraction[f]}
            else:
                s[a.from_to[f]][a.from_to[1 - f]] = a.fraction[f]

        self.tau: dict[Unit, dict[Task, int]] = {}  # processing time per batch of task i at unit u
        for u in self.units:
            self.tau[u] = {}
            for i in self.I_u[u]:
                self.tau[u][i] = i.batch_processing_time

    def generate_variables(self):
        self.variables: list[Variable] = []

        def new_var() -> Variable:
            res = Variable(len(self.variables))
            self.variables.append(res)
            return res

        self.MS = new_var()  # makespan
        self.Q: dict[Unit: dict[Task: dict[
            int, Variable]]]  # quantity of material undergoing processing of task i at unit u
        # at the beginning of period t (batch size)
        self.Q = {{u: {i: {t: new_var()
                           } for t in range(self.H)
                       } for i in self.tasks
                   } for u in self.units}

        self.p: dict[State: dict[int, Variable]]  # stock of product j in J_s at the end of period t (pj0 ˆ given)
        self.p = {{j: {t: new_var()
                       } for t in range(self.H)
                   } for j in self.J_s}

        self.x: dict[Unit: dict[Task: dict[int, Variable]]]  # = 1, if unit u starts processing task i at the beginning
        # of period t (0, otherwise)
        self.x = {{u: {i: {t: new_var()
                           } for t in range(self.H)
                       } for i in self.tasks
                   } for u in self.units}

    def generate_constraints(self):
        self.constraints: list[Constraint] = []

        def add_cons(l, r, o):
            self.constraints.append(Constraint(l, r, o))

        for t in range(self.H):
            # Makespan
            for u in self.U_e:
                for i in self.I_u[u]:
                    # t*x[uit]-MS <= 1-tau[ui]
                    l = {
                        self.MS: -1,
                        self.x[u][i][t]: t
                    }
                    r = 1 - self.tau[u][i]
                    add_cons(l, r, Operator.LEQ)

            # Batch size limits
            for u in self.U:
                for i in self.I_u[u]:
                    # B_min[u]*x[uit]-Q[uit] <= 0
                    l = {
                        self.x[u][i][t]: u.min_batch_size,
                        self.Q[u][i][t]: -1
                    }
                    r = 0
                    add_cons(l, r, Operator.LEQ)

                    # B_max[u]*x[uit]-Q[uit] >= 0
                    l = {
                        self.x[u][i][t]: u.max_batch_size,
                        self.Q[u][i][t]: -1
                    }
                    add_cons(l, r, Operator.GEQ)

            # Stock balance
            for j in self.J_s:
                # 0 = - p[jt] + p[j,t-1]+SUM[i in I_out[j]]SUM[u in U_i|t-tau[ui]>=1] Q[u,i,t-tau[ui]] -
                # - SUM[i in I_in[j]] alpha_in[ij] SUM
                l = {
                    self.p[j][t]: -1,
                    self.p[j][t - 1]: 1,

                }
                l +={{{
                            self.Q[u][i][t - self.tau[u][i]]: 1
                            if t - self.tau[u][i] >= 1 else None
                        } for u in self.U_i[i]
                    } for i in self.I_out[j]}
                l += {{{
                            self.Q[u][i][t]: self.alpha_in[i][j]
                        } for u in self.U_i[i]
                    } for i in self.I_in[j]}
                # TODO : check if += works correctly in here (else use l = {**l, **second_map}
                r = self.d[j][t] - self.e[j][t]
                add_cons(l, r, Operator.EQ)

            # Stock limits
            for j in self.J_s:
                # p[jt] <= P_max[j]
                l = {
                    self.p[j][t]: 1
                }
                r = j.max_stock
                add_cons(l, r, Operator.LEQ)

            # Production of non-storable goods

def test():
    states = [State()]
    tasks = [Task()]
    units = [Unit(tasks=tasks)]
    arcs = [Arc([states[0], tasks[0]])]
    stn = STN(states, tasks, arcs, units)
    stn.draw('0.dot')

    def make_state(x) -> State:
        if len(x) == 1:
            return State(name=x[0])
        if len(x) == 2:
            return State(name=x[0], max_stock=x[1])
        if len(x) == 3:
            return State(name=x[0], min_stock=x[1], max_stock=x[2])
        return State(x[0], x[1], x[2], x[3], x[4])

    def make_task(x) -> Task:
        match len(x):
            case 1:
                return Task(name=x[0])
            case 2:
                return Task(name=x[0], batch_processing_time=x[1])
            case 3:
                return Task(name=x[0], batch_processing_time=x[1], max_batch_size=x[2])
            case 4:
                return Task(x[0], x[1], x[2], x[3])  # TODO : fix the min max batch size (now they are fields of unit)

    def make_unit(x, tasks) -> Unit:
        return Unit(x[0], [tasks[i] for i in x[1]])  # TODO : fix the min max batch size (now they are fields of unit)

    def make_arc(x) -> Arc:
        match len(x):
            case 2:
                return Arc([x[0], x[1]])
            case 3:
                return Arc([x[0], x[1]], [x[2], x[2]])
            case 4:
                return Arc([x[0], x[1]], [x[2], x[3]])

    states = list(map(lambda x: make_state(x),
                      [
                          ["State 1"],
                          ["State 2"],
                          ["State 3"],
                          ["State 4", 100],
                          ["State 5", 200],
                          ["State 6", 0, 0, float('+inf'), True],
                          ["State 7", 150],
                          ["State 8", 0, 0, float('+inf'), True],
                          ["State 9"],
                          ["State 10"],
                      ]))
    tasks = list(map(lambda x: make_task(x),
                     [
                         ["Task 1/1", 1, 100],
                         ["Task 2/2", 2, 50],
                         ["Task 2/3", 2, 90],
                         ["Task 3/2", 2, 50],
                         ["Task 3/3", 2, 90],
                         ["Task 4/2", 1, 50],
                         ["Task 4/3", 1, 90],
                         ["Task 5/4", 1, 200],
                         ["Task 6/4", 2, 200],
                     ]))  # TODO : fix the min max batch size (now they are fields of unit)
    units = list(map(lambda x: make_unit(x, tasks),
                     [
                         ["Unit 1", [0]],
                         ["Unit 2", [1, 3, 5]],
                         ["Unit 3", [2, 4, 6]],
                         ["Unit 4", [7, 8]]
                     ]))  # TODO : fix the min max batch size (now they are fields of unit)
    arcs = list(map(lambda x: make_arc(x),
                    [
                        [states[0], tasks[0]],
                        [tasks[0], states[3]],
                        [states[3], tasks[1], 0.4],
                        [states[3], tasks[2], 0.4],
                        [states[1], tasks[3], 0.5],
                        [states[1], tasks[4], 0.5],
                        [states[2], tasks[3], 0.5],
                        [states[2], tasks[4], 0.5],
                        [states[2], tasks[5], 0.2],
                        [states[2], tasks[6], 0.2],
                        [tasks[1], states[8], 0.4],
                        [tasks[2], states[8], 0.4],
                        [tasks[1], states[6], 0.6],
                        [tasks[2], states[6], 0.6],
                        [tasks[3], states[4]],
                        [tasks[4], states[4]],
                        [tasks[5], states[5]],
                        [tasks[6], states[5]],
                        [states[6], tasks[5], 0.8],
                        [states[6], tasks[6], 0.8],
                        [states[4], tasks[1], 0.6],
                        [states[4], tasks[2], 0.6],
                        [states[5], tasks[7]],
                        [tasks[7], states[7], 0.1],
                        [states[7], tasks[8]],
                        [tasks[7], states[9], 0.9]
                    ]))
    stn = STN(states, tasks, arcs, units)
    stn.draw('1.dot')

    sp = SP(states, tasks, arcs, units, {states[8]: [1, 1, 1, 1, 1],
                                         states[9]: [1, 1, 1, 1, 1]},
            {states[0]: [1000, 1000, 1000, 1000, 1000],
             states[1]: [1000, 1000, 1000, 1000, 1000],
             states[2]: [1000, 1000, 1000, 1000, 1000],
             })
    sp.generate_dictionaries()
