# Scheduling_Problem solver

Постановку задачи можно прочитать [тут](scheduling_problem.pdf)

### class STN

```python
class Unit
class State
class Task
class Arc
```

Классы для представления частей химического завода

```python
class STN
```

Класс представляющий химический завод

```python
STN.draw('filename.dot')
```

Метод позволяющий сгенерировать '.dot' файл, который рендерится в графическое представление завода (примеры можно посмотреть в [pdf-ке](scheduling_problem.pdf) или в папке [tests/factory_diagrams/](../tests/factory_diagrams/))

> Готовые примеры фабрик: 
```python
import tests.factories

tests.factories.PrimitiveFactory
tests.factories.TwoUnitsFactory
tests.factories.ArticleExample1
```

### class SP

```python
class SP
```
Класс представляющий задачу о составлении расписаний

```python
factory = STN(...)
demand = dict(...)
supply = dict(...)
sp = SP(factory, demand, supply)
```

Чтобы создать экзмепляр класса нужно передать ему фабрику и списки спроса и предложения (пример есть в [main.py](../src/main.py))

```python
sp.solve(gt_max, ga_simple)
```

Метод ```solve(get_tableau, get_axis)``` вызывает сведение (приведенное в [pdf-ке](scheduling_problem.pdf)) и MILP солвер с эвристиками get_tableau, get_axis

Полученное решение хранится в полях ```SP```:

```python
sp.solution_value # -MS  
sp.solution       # список значений всех переменных
sp.q_dict_str()   # этот метод возвращает строковое представление того, какое количество материала, проходящего обработку задачи 𝑖 на химиче-ском реакторк 𝑢 в начале периода 𝑡 (размер партии)
```
