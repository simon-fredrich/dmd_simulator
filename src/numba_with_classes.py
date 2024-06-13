import numpy as np
from numba.experimental import jitclass
from numba import int32, float64
from numba.typed import List
from numba.types import ListType

spec = [("type", int32), ("val", float64[:]), ("result", float64)]


@jitclass(spec)
class First:
    def __init__(self):
        self.type = 1
        self.val = np.ones(100)
        self.result = 0.0

    def sum(self):
        self.result = np.sum(self.val)

print(First.class_type.instance_type)

spec1 = [("ListA", ListType(First.class_type.instance_type))]

@jitclass(spec1)
class Combined:
    def __init__(self):
        self.ListA = List((First(), First()))

    def sum(self):
        for i, c in enumerate(self.ListA):
            c.sum()

    def getresult(self):
        result = []
        for i, c in enumerate(self.ListA):
            result.append(c.result)
        return result


C = Combined()
C.sum()
result = C.getresult()
print(result)