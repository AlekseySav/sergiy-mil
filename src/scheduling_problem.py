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