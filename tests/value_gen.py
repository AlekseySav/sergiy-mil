import random


class ValueGenerator:
    mode: str
    x: float
    y: float

    def __init__(self, mode: str | None = 'normalvariate', E=0.0, D=1.0):
        self.mode = mode
        self.x = E
        self.y = D

    def value(self):
        match self.mode:
            case 'uniform':
                return random.uniform(self.x, self.y)
            case 'expovariate':
                return random.expovariate(1 / self.x)
            case 'betavariate':
                return random.betavariate(self.x, self.y)
            case 'normalvariate':
                return random.normalvariate(self.x, self.y ** 1/2)
            case _:
                return None
