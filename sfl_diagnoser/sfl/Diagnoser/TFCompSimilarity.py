__author__ = 'amir'

# from LightPSO import LightPSO
import operator
import functools
from.TF import TF
from functools import reduce
import math
from decimal import Decimal

class TFCompSimilarity(TF):
    def __init__(self, matrix, error, diagnosis, CompSimilarity, TestsSimilarity, ExperimentType, OriginalScorePercentage):
        super(TFCompSimilarity, self).__init__(matrix, error, diagnosis)
        self.CompSimilarity = CompSimilarity
        self.CompSimilarity_dict = dict()
        self.TestsSimilarity = TestsSimilarity
        self.ExperimentType = ExperimentType
        self.OriginalScorePercentage = OriginalScorePercentage
        

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

            # if self.ExperimentType == 'TestsSimilarity' or self.ExperimentType == 'BothSimilarities':
            #     div = self.TestsSimilarity[test_id]
            #     # div = (1 - self.TestsSimilarity[test_id])
            #     if div < 0.05:
            #         div = 0.05
            #     elif div > 0.95:
            #         div = 0.95
            #
            #     # div += 1
            #     # div = math.log(div)
            #     return original*( div)

            return original


        original_calc =  Decimal(reduce(operator.mul, list(map(lambda x: test_prob(*x), self.get_activity())), 1.0))
        parabolic_func = lambda original, similarity:  0.15*original**2 +0.15*original + 0.35*similarity**2 + 0.35*similarity
        sigmuid_func = lambda x: 1-math.e**(-((x/0.6)**4))


        norm1 = lambda x: (4*x) - 2 # norm values from [0,1] to [-2 , 2]
        sigmuid_func2 = lambda x: (Decimal(math.e)**(x) - Decimal(math.e)**(-x))/(Decimal(math.e)**(x) + Decimal(math.e)**(-x))
        norm2 = lambda x: (x+1)/2 # norm values from [-1,1] to [0 , 1]
        linear_func = lambda sim,original: Decimal((1.0 - self.OriginalScorePercentage) * sim) + Decimal(self.OriginalScorePercentage) * original
        avg_similarity = 0


        if self.ExperimentType == 'BothSimilarities': # old method with mul each sim
            #original_calc *= avg_similarity
            for i in range(len(self.diagnosis)):
                original_calc *= self.CompSimilarity[i]


        if self.ExperimentType == 'CompSimilarity': # new method with sigmoid
            for i in range(len(self.diagnosis)):
                avg_similarity += (self.CompSimilarity[i] / len(self.diagnosis))
            #original_calc = linear_func(avg_similarity, original_calc)
            #original_calc = norm2(sigmuid_func2(norm1(linear_func(avg_similarity, original_calc))))
            original_calc = linear_func(avg_similarity, original_calc)
            original_calc = norm1(original_calc)
            original_calc = sigmuid_func2(original_calc)
            original_calc = norm2(original_calc)
            original_calc = float(original_calc)

        # if self.ExperimentType == 'CompSimilarity': # new method with sigmoid
        #     for i in range(len(self.diagnosis)):
        #         avg_similarity += (self.CompSimilarity[i] / len(self.diagnosis))
        #     original_calc = sigmuid_func2(parabolic_func(original_calc, avg_similarity))



        # if original_calc > 0.99:
        #     original_calc = 0.99
        # elif original_calc < 0.01:
        #     original_calc = 0.01
        # avg = 0
        # if self.ExperimentType == 'CompSimilarity' or self.ExperimentType == 'BothSimilarities' :
        #     for i in range(len(self.diagnosis)):
        #         #original_calc *= (self.CompSimilarity[i] ** self.get_number_of_repetioions(self.diagnosis[i]))
        #         #original_calc *= (self.CompSimilarity[i])
        #         avg += (self.CompSimilarity[i] / len(self.diagnosis))
        #
        #     if avg == 1:
        #         original_calc /= 0.01
        #
        #     elif avg == 0:
        #          original_calc /= (1./0.01)
        #
        #     elif avg >= 0.5:
        #         original_calc /= (1.-avg)
        #     else:
        #         original_calc /= (1./avg)
        #
        # if original_calc > 1:
        #     original_calc = 1
        # elif original_calc < 0.01:
        #     original_calc = 0.01

        return original_calc





    def get_number_of_repetioions(self, comp_index):
        counter = 0
        for test in self.activity:
            counter += test[1][comp_index]
        return counter
