from enum import Enum

import pydot

from problem import Problem

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

INF = 1000000


class State:
    def __init__(self, name="state0", min_stock=0, max_stock=INF, ns=False):
        self.name = name
        self.min_stock = min_stock
        self.max_stock = max_stock
        self.ns = ns


class Task:
    def __init__(self, name="task0", batch_processing_time=0):
        self.name = name
        self.batch_processing_time = batch_processing_time


class Arc:
    def __init__(self, from_to: tuple[State, Task] | tuple[Task, State], fraction=None):
        if fraction is None:
            fraction = [1.0, 1.0]
        self.from_to = from_to
        self.fraction = fraction


class Unit:
    def __init__(self, name="unit0", tasks=None, min_batch_size=0, max_batch_size=INF):
        if tasks is None:
            tasks = []
        self.name = name
        self.tasks = tasks
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size


class Variable:
    def __init__(self, i: int):
        self.index = i


class Operator(Enum):
    LEQ = 1  # <=
    GEQ = 2  # >=
    EQ = 3  # ==


class Constraint:
    def __init__(self, left: dict[Variable, float], right: float = 0, operator: Operator = Operator.EQ):
        self.left = left
        self.right = right
        self.operator = operator

    def to_arrays(self, variables_count: int) -> list[list[float]]:
        if self.operator == Operator.EQ:
            self.operator = Operator.LEQ
            res = [self.to_arrays(variables_count)[0]]
            self.operator = Operator.GEQ
            res.append(self.to_arrays(variables_count)[0])
            return res
        res = [0.0 for _ in range(variables_count + 1)]
        for key in self.left.keys():
            res[key.index] = self.left[key] * (1 if self.operator == Operator.LEQ else -1)
        res[-1] = self.right * (1 if self.operator == Operator.LEQ else -1)
        return [res]


class STN:
    """
    State-Task-Network
    """

    def __init__(self, states=None, tasks=None, arcs=None, units=None,
                 other=None):
        if units is None:
            units = []
        if arcs is None:
            arcs = []
        if tasks is None:
            tasks = []
        if states is None:
            states = []
        if other is not None:
            self.states = other.states
            self.tasks = other.tasks
            self.arcs = other.arcs
            self.units = other.units
        else:
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
                                   label=f"{t.name}\n({t.batch_processing_time})")
            subgraphs[tasks_to_sg[t]].add_node(task_node)
        for s in subgraphs.values():
            graph.add_subgraph(s)
        for s in self.states:
            graph.add_node(pydot.Node(s.name, shape="diamond",
                                      label=f"{s.name}\n({s.min_stock},{s.max_stock})",
                                      color=("red" if s.ns else "black")))
        for a in self.arcs:
            graph.add_edge(pydot.Edge(a.from_to[0].name, a.from_to[1].name,
                                      label=(f"{a.fraction[0]}/{a.fraction[1]}" if a.fraction[0] != a.fraction[1]
                                             else f"{a.fraction[0] if a.fraction[0] != 1 else ''}")))
        return graph.write(filename)


class SP(STN):

    def __init__(self, sup: STN,
                 d=None, e=None):
        super().__init__(other=sup)
        self.problem = None
        if e is None:
            e = {}
        if d is None:
            d = {}
        self.U_e = None
        self.U_i = None
        self.U = None
        self.H = None
        self.J_ns = None
        self.J_s = None
        self.I_in = None
        self.I_u = None
        self.d = None
        self.e = None
        self.tau = None
        self.alpha_in = None
        self.alpha_out = None
        self.variables = None
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

    def _generate_dictionaries(self):
        self.I_u: dict[Unit, set[Task]] = {u: set() for u in self.units}  # tasks that can be performed by unit u
        for u in self.units:
            for t in u.tasks:
                self.I_u[u].add(t)

        self.I_in: dict[State, set[Task]] = {j: set() for j in self.states}  # tasks that use product [State] j as input
        for a in self.arcs:
            if isinstance(a.from_to[0], State):
                self.I_in[a.from_to[0]].add(a.from_to[1])

        self.I_out: dict[State, set[Task]] = {j: set() for j in self.states}  # tasks that produce product j
        for a in self.arcs:
            if isinstance(a.from_to[0], Task):
                self.I_out[a.from_to[1]].add(a.from_to[0])

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

        self.U_i: dict[Task, set[Unit]] = {i: set() for i in self.tasks}  # units capable of performing task i
        for u in self.units:
            for t in u.tasks:
                self.U_i[t].add(u)

        j_e: set[State] = set()  # products with external demand
        for s in self.d.keys():
            j_e.add(s)

        self.U_e: set[Unit] = set()  # units producing products with external demand
        for a in self.arcs:
            if isinstance(a.from_to[0], Task):
                if a.from_to[1] in j_e:
                    for u in self.U_i[a.from_to[0]]:
                        self.U_e.add(u)

        self.alpha_in: dict[Task, dict[State, float]] = {i: dict() for i in self.tasks}
        self.alpha_out: dict[
            Task, dict[State, float]] = {i: dict() for i in
                                         self.tasks}  # fixed proportion of input and output of product j in task i,
        # respectively
        for a in self.arcs:
            f = 1
            s = self.alpha_in
            if isinstance(a.from_to[0], Task):
                f = 0
                s = self.alpha_out
            s[a.from_to[f]][a.from_to[1 - f]] = a.fraction[f]

        self.tau: dict[Unit, dict[Task, int]] = {u: dict() for u in self.units}
        # processing time per batch of task i at unit u
        for u in self.units:
            self.tau[u] = {}
            for i in self.I_u[u]:
                self.tau[u][i] = i.batch_processing_time

        for j in self.states:
            if j not in self.d.keys():
                self.d[j] = [0 for _ in range(self.H)]
            if j not in self.e.keys():
                self.e[j] = [0 for _ in range(self.H)]

    def _generate_variables(self):
        self.variables: list[Variable] = []

        def new_var() -> Variable:
            res = Variable(len(self.variables))
            self.variables.append(res)
            return res

        self.MS = new_var()  # makespan
        self.Q: dict[Unit: dict[Task: dict[
            int, Variable]]]  # quantity of material undergoing processing of task i at unit u
        # at the beginning of period t (batch size)
        self.Q = {u: {i: {t: new_var()
                          for t in range(self.H)}
                      for i in self.tasks}
                  for u in self.units}

        self.p: dict[State: dict[int, Variable]]  # stock of product j in J_s at the end of period t (pj0 ˆ given)
        self.p = {j: {t: new_var()
                      for t in range(self.H)
                      } for j in self.J_s}

        self.x: dict[Unit: dict[Task: dict[int, Variable]]]  # = 1, if unit u starts processing task i at the beginning
        # of period t (0, otherwise)
        self.x = {u: {i: {t: new_var()
                          for t in range(self.H)
                          } for i in self.tasks
                      } for u in self.units}

    def _generate_constraints(self):
        self.constraints: list[Constraint] = []

        def add_cons(left, right, o):
            self.constraints.append(Constraint(left, right, o))

        for t in range(self.H):
            # Makespan
            for u in self.U_e:
                for i in self.I_u[u]:
                    l_hand = {
                        self.MS: -1,
                        self.x[u][i][t]: t
                    }
                    r = -self.tau[u][i]
                    add_cons(l_hand, r, Operator.LEQ)

            # Batch size limits
            for u in self.U:
                for i in self.I_u[u]:
                    l_hand = {
                        self.x[u][i][t]: u.min_batch_size,
                        self.Q[u][i][t]: -1
                    }
                    r = 0
                    add_cons(l_hand, r, Operator.LEQ)

                    # B_max[u]*x[uit]-Q[uit] >= 0
                    l_hand = {
                        self.x[u][i][t]: u.max_batch_size,
                        self.Q[u][i][t]: -1
                    }
                    add_cons(l_hand, r, Operator.GEQ)

            # Stock balance
            for j in self.J_s:
                l_hand = {
                    self.p[j][t]: -1,
                }
                if t >= 1:
                    l_hand[self.p[j][t - 1]] = 1
                for i in self.I_out[j]:
                    for u in self.U_i[i]:
                        if t - self.tau[u][i] >= 0:
                            l_hand[self.Q[u][i][t - self.tau[u][i]]] = self.alpha_out[i][j]
                for i in self.I_in[j]:
                    for u in self.U_i[i]:
                        l_hand[self.Q[u][i][t]] = -self.alpha_in[i][j]

                r = self.d[j][t] - self.e[j][t]
                add_cons(l_hand, r, Operator.EQ)

            # Stock limits
            for j in self.J_s:
                l_hand = {
                    self.p[j][t]: 1
                }
                r = j.max_stock
                add_cons(l_hand, r, Operator.LEQ)

            # Production of non-storable goods
            for j in self.J_ns:
                l_hand = dict()
                for i in self.I_out[j]:
                    for u in self.U_i[i]:
                        if t - self.tau[u][i] >= 0:
                            l_hand[self.Q[u][i][t - self.tau[u][i]]] = self.alpha_out[i][j]

                for i in self.I_in[j]:
                    for u in self.U_i[i]:
                        l_hand[self.Q[u][i][t]] = -self.alpha_in[i][j]

                r = 0
                add_cons(l_hand, r, Operator.EQ)

            # Assigning batches to production units
            for u in self.U:
                l_hand = dict()
                for i in self.I_u[u]:
                    if t >= self.tau[u][i]:
                        l_hand[self.x[u][i][t - self.tau[u][i]]] = 1
                r = 1
                add_cons(l_hand, r, Operator.LEQ)

            # Variable domains
            # Q[u][i][t] >= 0 - it is already done
            # p[j][t] >= 0 - it is also done
            # x[u][i][t] in {0, 1}
            for u in self.U:
                for i in self.I_u[u]:
                    l_hand = {
                        self.x[u][i][t]: 1
                    }
                    r = 1
                    add_cons(l_hand, r, Operator.LEQ)
                    r = 0
                    add_cons(l_hand, r, Operator.GEQ)

    def generate_problem(self):
        self._generate_dictionaries()
        self._generate_variables()
        self._generate_constraints()

        vars_count = len(self.variables)
        constraint_coeffs: list[list[float]] = []
        bounds: list[float] = []

        for cons in self.constraints:
            arrs = cons.to_arrays(vars_count)
            for arr in arrs:
                constraint_coeffs.append(arr[:-1])
                bounds.append(arr[-1])

        obj_coeffs = [0 for _ in range(vars_count)]
        obj_coeffs[self.MS.index] = -1

        type_constraints = [False for _ in range(vars_count)]
        for u in self.U:
            for i in self.I_u[u]:
                for t in range(self.H):
                    type_constraints[self.x[u][i][t].index] = True

        self.problem = Problem(constraint_coeffs, bounds, obj_coeffs, type_constraints)
