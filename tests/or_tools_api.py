from ortools.linear_solver import pywraplp
from problem import Problem


# sample code from Google or-tools tutorial
def solve_or_tools(p: Problem, solver: str = 'SCIP', result_values: list | None = None) -> tuple[float | None, str]:
    data = p.to_dict()
    output = "\nOR-TOOLS output:\n"
    # Create the mip solver with the SCIP backend.
    # SCIP for MILP problem, CLP for LP problem
    solver = pywraplp.Solver.CreateSolver(solver)
    if not solver:
        return None

    infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        if p.type_constraints[j]:
            x[j] = solver.IntVar(0, infinity, f'x[{j}]')
        else:
            x[j] = solver.NumVar(0, infinity, f'x[{j}]')
    output += f'Number of variables = {solver.NumVariables()}\n'

    for i in range(data['num_constraints']):
        constraint = solver.RowConstraint(-infinity, data['bounds'][i], '')
        for j in range(data['num_vars']):
            constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])
    output += f'Number of constraints = {solver.NumConstraints()}\n'
    # In Python, you can also set the constraints as follows.
    # for i in range(data['num_constraints']):
    #  constraint_expr = \
    # [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
    #  solver.Add(sum(constraint_expr) <= data['bounds'][i])

    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMaximization()
    # In Python, you can also set the objective as follows.
    # obj_expr = [data['obj_coeffs'][j] * x[j] for j in range(data['num_vars'])]
    # solver.Maximize(solver.Sum(obj_expr))

    output += '\n'
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        for j in range(data['num_vars']):
            if result_values is not None:
                result_values.append(x[j].solution_value())
            output += f"{x[j].name()} = {x[j].solution_value()}\n"
        output += f'solution value = {solver.Objective().Value()}\n\n'
        output += f'Problem solved in {solver.wall_time()} milliseconds\n'
        output += f'Problem solved in {solver.iterations()} iterations\n'
        output += f'Problem solved in {solver.nodes()} branch-and-bound nodes\n'
        return solver.Objective().Value(), output
    else:
        output += 'The problem does not have an optimal solution.\n'
        return None, output
