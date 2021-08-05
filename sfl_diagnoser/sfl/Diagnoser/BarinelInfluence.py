__author__ = 'amir'

import csv
import math
import sys
from .Barinel import Barinel
from .TFInfluence import TFInfluence


class BarinelInfluence(Barinel):

    def __init__(self):
        super(BarinelInfluence, self).__init__()
        self.influence_matrix = dict()
        self.influence_alpha = 0

    def set_influence_matrix(self, matrix):
        self.influence_matrix = matrix

    def set_influence_alpha(self, alpha):
        self.influence_alpha = alpha

    def tf_for_diag(self, diagnosis):
        return TFInfluence(self.get_matrix(), self.get_error(), diagnosis, self.influence_matrix, self.influence_alpha)
