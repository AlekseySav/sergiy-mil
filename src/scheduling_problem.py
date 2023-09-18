import pydot

"""
The problem is represented in this article: https://doi.org/10.1080/002075400189004

The problem is to minimize time needed to produce required amounts of different chemicals (products).
The products can be produced on Units, using via different processes (for example: heating, mixing) and other chemicals.

The input schema consists of three types of entities:
 - States - represent the chemicals - inputs and outputs from Units and their Tasks;
 - Tasks - represent different chemicals processes, taking materials from states as inputs
 and producing product's states as outputs;
 - Arcs - connect states and tasks in which they can be used as an input and connecting tasks with their outputs.

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
    batch_processing_time: float
    min_batch_size: float
    max_batch_size: float

    def __init__(self, name="task0", batch_processing_time=0, min_batch_size=0, max_batch_size=float('+inf')):
        self.name = name
        self.batch_processing_time = batch_processing_time
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size


class Arc:
    from_to: tuple[State, Task] | tuple[Task, State]
    fraction: list[float, float]

    def __init__(self, from_to: tuple[State, Task] | tuple[Task, State], fraction=[1.0, 1.0]):
        self.from_to = from_to
        self.fraction = fraction


class Unit:
    name: str
    tasks: list[Task]

    def __init__(self, name="unit0", tasks=[]):
        self.name = name
        self.tasks = tasks


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
                return Task(x[0], x[1], x[2], x[3])

    def make_unit(x, tasks) -> Unit:
        return Unit(x[0], [tasks[i] for i in x[1]])

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
                     ]))
    units = list(map(lambda x: make_unit(x, tasks),
                     [
                         ["Unit 1", [0]],
                         ["Unit 2", [1, 3, 5]],
                         ["Unit 3", [2, 4, 6]],
                         ["Unit 4", [7, 8]]
                     ]))
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
