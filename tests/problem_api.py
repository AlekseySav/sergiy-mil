class Problem:
    constraint_coeffs: list[list[float]]
    bounds: list[float]
    obj_coeffs: list[float]
    num_vars: int
    num_constraints: int

    def __init__(self, constraint_coeffs: list[list[float]], bounds: list[float], obj_coeffs: list[float]):
        self.constraint_coeffs = constraint_coeffs
        self.bounds = bounds
        self.obj_coeffs = obj_coeffs
        self.num_vars = len(constraint_coeffs[0])
        self.num_constraints = len(bounds)

    def to_dict(self) -> dict:
        """Stores the data for the problem."""
        data = {'constraint_coeffs': self.constraint_coeffs, 'bounds': self.bounds, 'obj_coeffs': self.obj_coeffs,
                'num_vars': self.num_vars, 'num_constraints': self.num_constraints}
        return data
