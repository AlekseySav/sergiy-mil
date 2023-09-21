# MILP solver

### solve_milp

```python
def solve_milp(problem: Tableau, constraints: list[bool],
               get_tableau: func_get_tableau,
               get_axis: func_get_axis,
               it_limit: int = 1000,
               bb_limit: int = 1000,
               solution: list[float] | None = None) -> tuple[float | None, str]
```

Солвер принимает 
 - задачу ```problem``` в виде экземпляра класса ```Tableau``` 
 - список ограничений на целочисленность переменных ```constraints```
 - эвристики выбора для алгоритма branch&bound: ```get_tableau``` и ```get_axis```
 - можно передать лимит на число итераций и ветвлений алгоритма (```it_limit```, ```bb_limit```)
 - пустой лист ```solution```, в который алгоритм запишет значения переменных полученного решения
Солвер возвращает пару значений
 - ```[float | None]``` - полученное оптимальное значение целевой функции (```None``` - если задача не имеет оптимального решения, т.е. overbounded или unbounded)
 - ```str``` - вывод алгоритма (количество итераций, ветвлений, время затраченное на решение задачи)

### Эвристики выбора для branch&bound

```python
func_get_tableau = Callable[[list[Tableau], list[bool]], Tableau | None]
func_get_axis = Callable[[Tableau, list[bool]], int | None]
```

- func_get_tableau - выбирает подзадачу из списка, для следующей итерации алгоритма
- func_get_axis - выбирает ось (целочисленную переменную), по которой надо разделить текущую подзадачу (т.е. сделать branch&bound)

```Python
def gt_simple(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
def ga_simple(problem: Tableau, in_constraints: list[bool]) -> int | None:
def gt_min(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
def gt_max(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
def gt_fifo(problems: list[Tableau], axis: list[bool]) -> Tableau | None:
```

- gt_simple - выбирает первую таблицу (т.е. обход дерева subdivisons будет как в BFS)
- ga_simple - выбирает первую подходящую ось (т.е. на этой оси есть ограничение на целочисленность и текущее значение переменной под него не попадает)
- gt_min - выбирает таблицу с мин. значением целевой ф-ии
- gt_max - выбирает таблицу с макс. значением целевой ф-ии
- gt_fifo - выбирает последнюю таблицу (т.е. алгоритм будет максимально углубляться, как в DFS)
