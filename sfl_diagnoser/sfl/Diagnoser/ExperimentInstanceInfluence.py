import copy
import math
import random
from math import ceil
from .dynamicSpectrumInfluence import DynamicSpectrumInfluence
from .Experiment_Data import Experiment_Data
import numpy
from .ExperimentInstance import ExperimentInstance

TERMINAL_PROB = 0.7


class ExperimentInstanceInfluence(ExperimentInstance):
    def __init__(self, initial_tests, error, priors, bugs, pool, components, estimated_pool=None, **kwargs):
        super(ExperimentInstanceInfluence, self).__init__(initial_tests, error, priors, bugs, pool, components, estimated_pool, **kwargs)
        """ check the influence matrix and influence alpha"""
        assert  0.0 <= Experiment_Data().influence_alpha <= 1.0 , "wrong value of influence_alpha"
        tests = Experiment_Data().POOL.keys()
        assert len(Experiment_Data().influence_matrix.keys()) == len(tests) , "different number of tests in influence matrix"
        assert sorted(Experiment_Data().influence_matrix.keys()) == sorted(tests) , "different tests in influence matrix"
        for test in tests:
            assert len(set(Experiment_Data().POOL[test])) == len(Experiment_Data().POOL[test]), \
                "trace of test {0} is faulty".format(test)
            assert len(Experiment_Data().influence_matrix[test]) == len(Experiment_Data().POOL[test]), \
                "influence of test {0} is faulty".format(test)
            assert all(list(map(lambda c: 0.0 <= c <= 1.0, Experiment_Data().influence_matrix[test].values()))), \
                "wrong value of influence information of test {0}".format(test)

    def initials_to_DS(self):
        ds = DynamicSpectrumInfluence()
        ds.setTestsComponents(copy.deepcopy([Experiment_Data().POOL[test] for test in self.get_initials()]))
        ds.setprobabilities(list(self.priors))
        ds.seterror([self.get_error()[test] for test in self.get_initials()])
        ds.settests_names(list(self.get_initials()))
        influence_matrix = []
        for test in self.get_initials():
            t_dict = {}
            for component, data in Experiment_Data().influence_matrix[test].items():
                id = Experiment_Data().get_component_id(component)
                t_dict[id] = data
            influence_matrix.append(t_dict)
        ds.set_influence_matrix(influence_matrix)
        ds.set_influence_alpha(Experiment_Data().influence_alpha)
        return ds
