from tableau import Tableau, array

'''
Problem:
    Min/Max 7x+6y, where
        2x + 4y <= 16
        3x + 2y <= 12
        x, y >= 0

Setup:
    t = Tableau(func=array([7, 6]))
    t.add_constraint(array([2, 4, 16]))            # 2x + 4y <= 16
    t.add_constraint(array([3, 2, 0, 12]))         # 3x + 2y <= 12
                            ^~~~~~~~ s1
Result after setup:
    func        7  6  0  0
    matrix      2  4  1  0  16
                3  2  0  1  12
    basis             ^  ^
    count=4
'''


def test_split_subdivision():
    t = Tableau(func=array([7, 6]))
    t.add_constraint(array([2, 4, 16]))  # 2x + 4y <= 16
    t.add_constraint(array([3, 2, 0, 12]))  # 3x + 2y <= 12
    t = Tableau.from_matrix