from .ExperimentInstance import ExperimentInstance
from .ExperimentInstanceCompSimilarity import ExperimentInstanceCompSimilarity
from .ExperimentInstanceOptimize import ExperimentInstanceOptimize
from .Experiment_Data import Experiment_Data


class ExperimentInstanceFactory(object):

    @staticmethod
    def create_key(initial_tests, error):
        return repr(sorted(initial_tests)) + "-" + repr(
            sorted(list(map(lambda x: x[0], filter(lambda x: x[1] == 1, error.items())))))

    @staticmethod
    def get_experiment_instance(initials, error, priors, bugs, pool, components, estimated_pool, experiment_type, **kwargs):
        classes = {'normal': ExperimentInstance,
                   'CompSimilarity': ExperimentInstanceCompSimilarity,
                   'TestsSimilarity' : ExperimentInstanceCompSimilarity,
                   'BothSimilarities' : ExperimentInstanceCompSimilarity}
                   #, 'optimize': ExperimentInstanceOptimize.Instances_Management().get_instance}
        return classes.get(experiment_type, classes[experiment_type])(initials, error, priors, bugs, pool, components, estimated_pool, **kwargs)