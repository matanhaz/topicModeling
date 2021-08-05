from .BarinelOptimize import BarinelOptimize
from .FullMatrix import FullMatrix


class FullMatrixOptimize(FullMatrix):
    def __init__(self):
        super(FullMatrixOptimize, self).__init__()

    def _create_barinel(self):
        return BarinelOptimize()

    def _create_full_matirx(self):
        return FullMatrixOptimize()
