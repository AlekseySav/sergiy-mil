from or_tools_api import solve_or_tools
import tabular_lp as lp
import numpy as np
import random
from problem_gen import Problem, ProblemGenerator
from test_milp import comp


'''
test tableau methods
'''

def test_tableau():
    lhs = lp.array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    rhs = lp.array([1, 3, 5])
    obj = lp.array([4, 5, 0])
    t = lp.Tableau.make(lhs, rhs, obj, [0, 1, 2])
    assert t._basis == [0, 1, 2, 3]
    assert np.allclose(t._matrix, lp.array([
        [1, -4, -5, 0, 0],
        [0,  1,  2, 3, 1],
        [0,  4,  5, 6, 3],
        [0,  7,  8, 9, 5]
    ]))
    p = lp.Tableau.from_matrix(lp.array([
        [1, 2, 3, 1],
        [4, 5, 6, 3],
        [7, 8, 9, 5]
    ]), [0, 1, 2], lp.array([4, 5, 0]))
    assert t == p
    t.add_constraint(lp.array([1, 5, 3]), lp.Float(8), lp.ConstraintSign.GEQ)
    assert t._basis == [0, 1, 2, 3, 4]
    assert np.allclose(t._matrix, lp.array([
        [1, -4, -5,  0,  0,  0],
        [0,  1,  2,  3,  0,  1],
        [0,  4,  5,  6,  0,  3],
        [0,  7,  8,  9,  0,  5],
        [0, -1, -5, -3,  1, -8]
    ]))
    assert t.variables_count == 4
    assert t.solution()[0] == 0
    assert np.allclose(t.solution()[1], lp.array([1, 3, 5, -8]))
    t.add_restriction(0, lp.Float(8), lp.ConstraintSign.LEQ)
    assert np.allclose(t._matrix, lp.array([
        [1, -4, -5,  0,  0,  0,  0],
        [0,  1,  2,  3,  0,  0,  1],
        [0,  4,  5,  6,  0,  0,  3],
        [0,  7,  8,  9,  0,  0,  5],
        [0, -1, -5, -3,  1,  0, -8],
        [0,  0, -2, -3,  0,  1,  7]
    ]))

'''
basic tests here
'''

def is_optimal_feasible(t: lp.Tableau):
    res = True
    rhs = t._matrix[:, -1] # last column = values
    obj = t._matrix[0, :] # first row = objective function
    print(rhs, obj, t._basis)
    res = res and all(rhs[1:] >= 0) # assert feasibility
    res = res and all(obj[:-1] >= 0) # assert optimality
    res = res and all(obj[t._basis[1:]] == 0) # assert t._basis is correct
    solution = np.zeros_like(t._matrix[0])
    solution[t._basis] = rhs # get solution
    return res


def test_primal_basic():
    t = lp.Tableau(lp.array([
        [1, -8, -7, 0, 0, 0],
        [0, -3, -5, 1, 0, 0],
        [0,  9,  4, 0, 1, 133]
    ]), [0, 3, 4])
    assert not is_optimal_feasible(t)
    lp.run_primal_simplex(t)

    t = lp.Tableau(lp.array([
        [1, -8, -5, 0, 0, 0],
        [0,  5,  4, 1, 0, 97],
        [0,  2,  8, 0, 1, 0],
    ]), [0, 3, 4])
    assert not is_optimal_feasible(t)
    lp.run_primal_simplex(t)
    assert t.solution()[0] == 0

    # https://web.stanford.edu/class/msande310/lecture09.pdf
    t = lp.Tableau(lp.array([
        [1, -100, -10, -1, 0, 0, 0,     0],
        [0,    1,   0,  0, 1, 0, 0,     1],
        [0,   20,   1,  0, 0, 1, 0,   100],
        [0,  200,  20,  1, 0, 0, 1, 10000],
    ]), [0, 4, 5, 6])
    assert not is_optimal_feasible(t)
    lp.run_primal_simplex(t)
    assert t._basis == [0, 4, 5, 3]
    assert np.allclose(t._matrix, lp.array([
        [1, 100, 10, 0, 0, 0, 1, 10000],
        [0,   1,  0, 0, 1, 0, 0,     1],
        [0,  20,  1, 0, 0, 1, 0,   100],
        [0, 200, 20, 1, 0, 0, 1, 10000]
    ]))
    assert is_optimal_feasible(t)

    # https://www.youtube.com/watch?v=FY97HLnstVw&ab_channel=GOALPROJECT
    t = lp.Tableau(lp.array([
        [1, -7, -4, 0, 0, 0,  0],
        [0,  2,  1, 1, 0, 0, 20],
        [0,  1,  1, 0, 1, 0, 18],
        [0,  1,  0, 0, 0, 1,  8]
    ]), [0, 3, 4, 5])
    assert not is_optimal_feasible(t)
    lp.run_primal_simplex(t)
    assert t._basis == [0, 2, 5, 1]
    assert np.allclose(t._matrix, lp.array([
        [1, 0, 0,  3,  1, 0, 78],
        [0, 0, 1, -1,  2, 0, 16],
        [0, 0, 0, -1,  1, 1,  6],
        [0, 1, 0,  1, -1, 0,  2]
    ]))
    assert is_optimal_feasible(t)

    # https://youtu.be/rc9E1yLHFgo?si=VwKQRjOsU-HPooJz
    t = lp.Tableau(lp.array([
        [1, -6, -5, -4, 0, 0, 0,   0],
        [0,  2,  1,  1, 1, 0, 0, 240],
        [0,  1,  3,  2, 0, 1, 0, 360],
        [0,  2,  1,  2, 0, 0, 1, 300]
    ]), [0, 4, 5, 6])
    assert not is_optimal_feasible(t)
    lp.run_primal_simplex(t)
    assert t._basis == [0, 1, 2, 6]
    assert np.allclose(t._matrix, lp.array([
        [1, 0, 0, .2, 2.6,  .8, 0, 912],
        [0, 1, 0, .2,  .6, -.2, 0,  72],
        [0, 0, 1, .6, -.2,  .4, 0,  96],
        [0, 0, 0,  1,  -1,   0, 1,  60]
    ]))
    assert is_optimal_feasible(t)

    # https://youtu.be/pVWsXZh81IU?si=zE3_L7zTJzwPmYx3
    t = lp.Tableau(lp.array([
        [ 1, -1, -2, 0, 0, 0,   0],
        [ 0,  1,  1, 1, 0, 0,   3],
        [ 0,  0,  1, 0, 1, 0,   2],
        [ 0, .5,  1, 0, 0, 1, 2.5]
    ]), [0, 3, 4, 5])
    assert not is_optimal_feasible(t)
    lp.run_primal_simplex(t)
    assert t._basis == [0, 1, 2, 5]
    assert np.allclose(t._matrix, lp.array([
        [1, 0, 0,   1,   1, 0, 5],
        [0, 1, 0,   1,  -1, 0, 1],
        [0, 0, 1,   0,   1, 0, 2],
        [0, 0, 0, -.5, -.5, 1, 0]
    ]))
    assert is_optimal_feasible(t)


def test_dual_basic():
    t = lp.Tableau(lp.array([
        [1, 114, 165, 0, 0, 0],
        [0,  -6,   2, 1, 0, -18],
        [0,   7,   8, 0, 1,  64]
    ]), [0, 3, 4])
    assert not is_optimal_feasible(t)
    lp.run_dual_simplex(t)

    t = lp.Tableau(lp.array([
        [1, 97, 177, 0, 0, 0],
        [0,  2,  -7, 1, 0, -66],
        [0,  2,   2, 0, 1,  29]
    ]), [0, 3, 4])
    assert not is_optimal_feasible(t)
    lp.run_dual_simplex(t)
    assert is_optimal_feasible(t)

    # https://youtu.be/ORn1MVC2gq4?si=gwLWGsR5fTf0Eyqn
    t = lp.Tableau(lp.array([
        [1,  3,  4, 0, 0, 0, 0,    0],
        [0,  2,  1, 1, 0, 0, 0,  600],
        [0,  1,  1, 0, 1, 0, 0,  225],
        [0,  5,  4, 0, 0, 1, 0, 1000],
        [0, -1, -2, 0, 0, 0, 1, -150]
    ]), [0, 3, 4, 5, 6])
    assert not is_optimal_feasible(t)
    lp.run_dual_simplex(t)
    assert t._basis == [0, 3, 4, 5, 2]
    assert np.allclose(t._matrix, lp.array([
        [1,   1, 0, 0, 0, 0,   2, -300],
        [0, 1.5, 0, 1, 0, 0,  .5,  525],
        [0,  .5, 0, 0, 1, 0,  .5,  150],
        [0,   3, 0, 0, 0, 1,   2,  700],
        [0,  .5, 1, 0, 0, 0, -.5,   75]
    ]))
    assert is_optimal_feasible(t)

    # https://www.youtube.com/watch?v=39uebUF0VuU&ab_channel=StudyBuddy
    t = lp.Tableau(lp.array([
        [1,  3,  2, 0, 0, 0, 0,   0],
        [0, -1, -1, 1, 0, 0, 0,  -1],
        [0,  1,  1, 0, 1, 0, 0,   7],
        [0, -1, -2, 0, 0, 1, 0, -10],
        [0, -1,  1, 0, 0, 0, 1,   3]
    ]), [0, 3, 4, 5, 6])
    assert not is_optimal_feasible(t)
    lp.run_dual_simplex(t)
    assert is_optimal_feasible(t)
    assert t._basis == [0, 3, 4, 2, 1]

def test_milp_iter():
    t = lp.Tableau(_matrix=lp.array([[  1.        ,   0.        ,   0.        ,  -0.        ,
          0.88888889,  -0.        ,   1.66666667,   0.        ,
        207.88888889],
       [  0.        ,   0.        ,   1.        ,   0.        ,
         -0.11111111,   0.        ,   0.66666667,   0.        ,
         16.88888889],
       [  0.        ,   1.        ,   0.        ,   0.        ,
          0.        ,   0.        ,   1.        ,   0.        ,
         49.        ],
       [ -0.        ,  -0.        ,  -0.        ,   1.        ,
          1.        ,   0.        ,  -3.        ,   0.        ,
          3.        ],
       [ -0.        ,  -0.        ,  -0.        ,  -0.        ,
          0.11111111,   1.        ,  -0.66666667,   0.        ,
          0.11111111],
       [  0.        ,  -0.        ,  -0.        ,  -0.        ,
         -0.11111111,  -0.        ,   0.66666667,   1.        ,
         -0.11111111]]), _basis=[0, 2, 1, 3, 5, 7])   
    assert True == lp.run_dual_simplex(t)

def run_random_tests(iters, bounds, obj, m, n, method, check):
    gen = ProblemGenerator(
        {
            'constraints_coeffs': 'uniform',
            'bounds': 'uniform',
            'obj_coeffs': 'uniform'
        },
        {
            'constraints_coeffs': [-10, 10],
            'bounds': bounds,
            'obj_coeffs': obj
        },
        m,
        n
    )
    for _ in range(iters):
        p = gen.value()
        t = p.to_tableau()
        if not check(t):
            continue
        or_tools_res, _ = solve_or_tools(p)
        ok = method(t)
        assert (or_tools_res is None) == (ok == False), str(p)
        if or_tools_res is not None:
            assert comp(or_tools_res, t.solution()[0]), str(p)


def test_primal_simplex_random():
    check = lambda t: not all(t._matrix[0, :] >= 0)
    run_random_tests(100, [0, 3], [-3, 3], 3, 3, lp.run_primal_simplex, check)
    run_random_tests(1000, [0, 10], [-10, 10], 2, 2, lp.run_primal_simplex, check)
    run_random_tests(1000, [0, 200], [-100, 100], 2, 2, lp.run_primal_simplex, check)
    run_random_tests(100,  [0, 200], [-100, 100], 8, 8, lp.run_primal_simplex, check)
    run_random_tests(100,  [0, 200], [-100, 100], 30, 30, lp.run_primal_simplex, check)


def test_dual_simplex_random():
    check = lambda t: not all(t._matrix[:, -1] >= 0)
    run = lambda t: lp.run_dual_simplex(t) # and lp.run_primal_simplex(t)
    run_random_tests(100, [-3, 3], [-3, 0], 3, 3, run, check)
    run_random_tests(1000, [-10, 10], [-10, 0], 2, 2, run, check)
    run_random_tests(1000, [-100, 100], [-200, 0], 2, 2, run, check)
    run_random_tests(100, [-100, 100], [-200, 0], 10, 10, run, check)
    run_random_tests(100, [-100, 100], [-200, 0], 30, 30, run, check)

