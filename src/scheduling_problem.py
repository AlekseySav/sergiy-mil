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

        graph = pydot.Dot("STN-graph", graph_type="digraph", bgcolor="white")
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
                                      label=f"{a.fraction[0]}/{a.fraction[1]}", decorate="true"))
        return graph.write(filename)


def test():
    states = [State()]
    tasks = [Task()]
    units = [Unit(tasks=tasks)]
    arcs = [Arc([states[0], tasks[0]])]
    stn = STN(states, tasks, arcs, units)
    stn.draw('0.dot')

    states = [
        State(name="State 1"),
        State(name="State 2"),
        State(name="State 3"),
        State(name="State 4", max_stock=100),
        State(name="State 5", max_stock=200),
        State(name="State 6", ns=True),
        State(name="State 7", max_stock=150),
        State(name="State 8", ns=True),
        State(name="State 9"),
        State(name="State 10"),
    ]
    tasks = [
        Task(name="Task 1/1", batch_processing_time=1, max_batch_size=100),
        Task(name="Task 2", batch_processing_time=2, max_batch_size=50),
        Task(name="Task 2", batch_processing_time=2, max_batch_size=90),
        Task(name="Task 3", batch_processing_time=2, max_batch_size=50),
        Task(name="Task 3", batch_processing_time=2, max_batch_size=90),
        Task(name="Task 4", batch_processing_time=1, max_batch_size=50),
        Task(name="Task 4", batch_processing_time=1, max_batch_size=90),
        Task(name="Task 5", batch_processing_time=1, max_batch_size=200),
        Task(name="Task 6", batch_processing_time=2, max_batch_size=200),
    ]
    units = [
        Unit(name="Unit 1", tasks=[tasks[0]]),
        Unit(name="Unit 2", tasks=[tasks[1], tasks[3], tasks[5]]),
        Unit(name="Unit 3", tasks=[tasks[2], tasks[4], tasks[6]]),
        Unit(name="Unit 4", tasks=[tasks[7], tasks[8]]),
    ]
    arcs = [
        Arc([states[0], tasks[0]]),
        Arc([tasks[0], states[3]]),
        Arc([states[3], tasks[2]], fraction=[0.4, 0.4]),
        Arc([states[3], tasks[3]], fraction=[0.4, 0.4]),
    ]
    stn = STN(states, tasks, arcs, units)
    stn.draw('1.dot')
