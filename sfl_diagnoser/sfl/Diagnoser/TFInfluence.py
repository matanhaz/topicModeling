__author__ = 'amir'

# from LightPSO import LightPSO
import operator
import functools
from.TF import TF


class TFInfluence(TF):
    def __init__(self, matrix, error, diagnosis, influence_matrix, influence_alpha):
        super(TFInfluence, self).__init__(matrix, error, diagnosis)
        self.influence_matrix = influence_matrix
        self.influence_alpha = influence_alpha
        self.influence_dict = dict()
        for test_id, test in enumerate(self.influence_matrix):
            test_dict = dict(map(lambda c: (c, test[c]), self.get_active_components()[test_id]))
            self.influence_dict[test_id] = test_dict

    def maximize(self):
        max_value = super(TFInfluence, self).maximize()
        influence_probability = self.probabilty(self.influence_dict)
        return max_value * self.influence_alpha + influence_probability * (1 - self.influence_alpha)

    def calculate(self, values):
        return self.probabilty(values)
