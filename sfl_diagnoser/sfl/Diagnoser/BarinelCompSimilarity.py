__author__ = 'amir'

import csv
import math
import sys
from .Barinel import Barinel
from .TFCompSimilarity import TFCompSimilarity


class BarinelCompSimilarity(Barinel):

    def __init__(self):
        super(BarinelCompSimilarity, self).__init__()
        self.CompSimilarity = []
        self.TestsSimilarity = []
        self.ExperimentType = None
     

    def set_CompSimilarity(self, s):
        self.CompSimilarity = s
    def set_TestsSimilarity(self, s):
        self.TestsSimilarity = s
    def set_ExperimentType(self, t):
        self.ExperimentType = t


   

    def tf_for_diag(self, diagnosis):
        return TFCompSimilarity(self.get_matrix(), self.get_error(), diagnosis, list(map(self.CompSimilarity.get, diagnosis)), self.TestsSimilarity,self.ExperimentType)
