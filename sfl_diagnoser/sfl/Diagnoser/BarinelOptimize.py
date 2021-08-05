__author__ = 'amir'

import csv
import math
import sys
from .Barinel import Barinel

from .DiagnosisOptimize import DiagnosisOptimize
from .TFOptimize import TFOptimize


class BarinelOptimize(Barinel):

    def __init__(self):
        super(BarinelOptimize, self).__init__()

    def tf_for_diag(self, diagnosis):
        return TFOptimize(self.get_matrix(), self.get_error(), diagnosis)

    def _new_diagnosis(self, diagnosis):
        return DiagnosisOptimize(diagnosis)
