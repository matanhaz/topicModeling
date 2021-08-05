import copy
import math
import random
from math import ceil
from .dynamicSpectrumOptimize import dynamicSpectrumOptimize
from .Experiment_Data import Experiment_Data, Singleton
import numpy
from .ExperimentInstance import ExperimentInstance

TERMINAL_PROB = 0.7


class Instances_Management(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.instances = {}
        self.clear()

    def clear(self):
        self.instances.clear()
        self.instances = {}

    def get_instance(self, initial_tests, error):
        key = repr(sorted(initial_tests)) + "-" + repr(sorted(list(map(lambda x: x[0], filter(lambda x: x[1] == 1, error.items())))))
        if key not in self.instances:
            self.instances[key] = self.create_instance_from_key(key)
        return self.instances[key]

    def create_instance_from_key(self, key):
        initial, failed = key.split('-')
        error = dict([(i, 1 if i in eval(failed) else 0) for i in Experiment_Data().POOL])
        return ExperimentInstanceOptimize(eval(initial), error)


class ExperimentInstanceOptimize(ExperimentInstance):
    def __init__(self, initial_tests, error):
        super(ExperimentInstanceOptimize, self).__init__(initial_tests, error)

    def _create_ds(self):
        return dynamicSpectrumOptimize()

    def create_instance(self, initial_tests, error):
        return Instances_Management().get_instance(initial_tests, error)
