from copy import copy
from src.lp.lp import Variable
from src.lp.lp import Column
from src.lp.lp import Matrix
from fractions import Fraction
from math import ceil
from heuristic import Parser

class Partition:
    constraint_coeffs: Matrix
    bounds: Column
    obj_coeffs: Column
    epoch: int
    LR_res: Fraction
    LR_point: Column

    def __init__(self, matrix=[], bounds_column=[], obj_column=[]):
        self.constraint_coeffs = matrix
        self.bounds = bounds_column
        self.obj_coeffs = obj_column
        self.epoch = 0
        self.LR_res = Fraction(0)

    def run_lr(self):
        return ...

    def add_constraint(self, var_idx: int, bound: Fraction, more: bool):
        partition = copy(self)
        new_constraint = [Fraction(0) for _ in range(len(partition.constraint_coeffs[0]) + 1)]
        new_constraint[var_idx] = Fraction(1)
        # s[new] >= 0
        for i in range(len(partition.constraint_coeffs)):
            partition.constraint_coeffs[i].append(Fraction(0))
        # x[i] >= [bound] + 1
        # x[i] + s[new] = [bound] + 1
        if more:
            new_constraint[-1] = Fraction(1)
            partition.bounds.append(Fraction(ceil(bound) + 1))
        # x[i] <= [bound]
        # x[i] - s[new] = [bound]
        # -x[i] + s[new] = -[bound]
        else:
            new_constraint[-1] = Fraction(1)
            new_constraint[var_idx] = Fraction(-1)
            partition.bounds.append(-Fraction(ceil(bound)))
        partition.constraint_coeffs.append(new_constraint)
        return partition


class BranchBound:
    partitions: list[Partition]
    partitions_by_LP: dict[Fraction, Partition]
    partitions_by_order: dict[Fraction, Partition]
    partitions_by_size: dict[Fraction, Partition]
    parser: Parser

    def __init__(self):
        partitions = []
        partitions_by_LP: {}
        partitions_by_order: {}
        partitions_by_size: {}

    def split_partition(self, partition: Partition):
        partition += self.parser.split(partition)

    def add_partition(self):

