from .FullMatrixOptimize import FullMatrixOptimize
from .dynamicSpectrum import dynamicSpectrum
from functools import partial


class dynamicSpectrumOptimize(dynamicSpectrum):
    def __init__(self):
        super(dynamicSpectrumOptimize, self).__init__()

    def _create_full_matrix(self):
        return FullMatrixOptimize()
