__author__ = 'amir'

# from LightPSO import LightPSO
import operator
import functools
from.TF import TF
from functools import reduce
import math


class TFCompSimilarity(TF):
    def __init__(self, matrix, error, diagnosis, CompSimilarity, TestsSimilarity, ExperimentType):
        super(TFCompSimilarity, self).__init__(matrix, error, diagnosis)
        self.CompSimilarity = CompSimilarity
        self.CompSimilarity_dict = dict()
        self.TestsSimilarity = TestsSimilarity
        self.ExperimentType = ExperimentType
        

    def maximize(self):
        max_value = super(TFCompSimilarity, self).maximize()
        #CompSimilarity_probability = self.probabilty(self.CompSimilarity_dict)
        return max_value 

    def calculate(self, values):
        return self.probabilty(values)

    def probabilty(self, h_dict):
        # h_dict is dict of dicts for test to comps
        def test_prob(test_id, v, e):
            # if e==0 : h1*h2*h3..., if e==1: 1-h1*h2*h3...
            original = e + ((-2.0 * e + 1.0) * reduce(operator.mul,
                                                   list(map(h_dict[test_id].get, self.get_active_components()[test_id])), 1.0))

            if self.ExperimentType == 'TestsSimilarity' or self.ExperimentType == 'BothSimilarities':
                div = self.TestsSimilarity[test_id]
                # div = (1 - self.TestsSimilarity[test_id])
                if div < 0.05:
                    div = 0.05
                elif div > 0.95:
                    div = 0.95
                
                # div += 1 
                # div = math.log(div)
                return original*( div)
            return original


        original_calc =  reduce(operator.mul, list(map(lambda x: test_prob(*x), self.get_activity())), 1.0)
        if original_calc > 0.99:
            original_calc = 0.99
        elif original_calc < 0.01:
            original_calc = 0.01
        avg = 0
        if self.ExperimentType == 'CompSimilarity' or self.ExperimentType == 'BothSimilarities' :
            for i in range(len(self.diagnosis)):
                #original_calc *= (self.CompSimilarity[i] ** self.get_number_of_repetioions(self.diagnosis[i]))
                #original_calc *= (self.CompSimilarity[i])
                avg += (self.CompSimilarity[i] / len(self.diagnosis))

            if avg == 1:
                original_calc /= 0.01

            elif avg == 0:
                 original_calc /= (1./0.01)

            elif avg >= 0.5:
                original_calc /= (1.-avg)
            else:
                original_calc /= (1./avg)

        if original_calc > 1:
            original_calc = 1
        elif original_calc < 0.01:
            original_calc = 0.01

        return original_calc





    def get_number_of_repetioions(self, comp_index):
        counter = 0
        for test in self.activity:
            counter += test[1][comp_index]
        return counter
