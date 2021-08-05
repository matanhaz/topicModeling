__author__ = 'amir'

import csv
import functools
import gc
import glob
import os
import time
from threading import Thread
from numpy.random import choice

import sfl.Planner.lrtdp.LRTDPModule
import sfl.Planner.mcts.main
from sfl.Planner.mcts.mcts import mcts_uct, clear_states

import sfl.Diagnoser.diagnoserUtils
import sfl.Diagnoser.ExperimentInstance
from .Diagnoser.Diagnosis_Results import Diagnosis_Results


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception, e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception, je:
                print 'error starting thread'
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


class Metrics(object):
    def __init__(self, ei, step):
        ei.diagnose()
        self.metrics = {"step": step}
        diagnosis_res = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error)
        self.metrics.update(diagnosis_res.metrics)
        self.metrics["max_probability"] = ei.getMaxProb()
        self.metrics["#_diagnoses"] = len(ei.diagnoses)

    def get_metrics(self):
        return self.metrics

    def add_time(self, total_time):
        self.metrics["total_time"] = total_time


class AbstractPlanner(object):
    def __init__(self):
        self.partial_metrics = dict()
        self.metrics = None

    # @timeout(3600)
    def plan(self, ei):
        # .sfl.Diagnoser.ExperimentInstance.Instances_Management().clear()
        gc.collect()
        steps = 0
        start = time.time()
        self.partial_metrics[steps] = Metrics(ei, steps)
        while not self.stop_criteria(ei):
            gc.collect()
            ei = ei.addTests(self._plan(ei))
            steps += 1
            self.partial_metrics[steps] = Metrics(ei, steps).get_metrics()
        self.metrics = Metrics(ei, steps)
        self.metrics.add_time(time.time() - start)

    def stop_criteria(self, ei):
        PROBABILITY_STOP = 0.7
        ei.diagnose()
        return ei.getMaxProb() > PROBABILITY_STOP or len(ei.get_optionals_actions()) == 0

    def _plan(self, ei):
        return choice(ei.get_optionals_actions())

    def get_name(self):
        return self.__class__.__name__


class MCTSPlanner(AbstractPlanner):
    def __init__(self, approach, iterations):
        super(MCTSPlanner, self).__init__()
        self.approach, self.iterations = approach, iterations

    def plan(self, ei):
        clear_states()
        super(MCTSPlanner, self).plan(ei)

    def _plan(self, ei):
        action, weight = mcts_uct(ei, self.iterations, self.approach)
        return action

    def get_name(self):
        return "_".join(list(map(str, [self.__class__.__name__, self.approach, self.iterations])))


class InitialsPlanner(AbstractPlanner):
    def stop_criteria(self, ei):
        return True


class HPPlanner(AbstractPlanner):
    def _plan(self, ei):
        return ei.hp_next()


class LRTDPPlanner(AbstractPlanner):
    def __init__(self, approach="uniform", iterations=1, greedy_action_treshold=1, epsilon=0.001):
        super(LRTDPPlanner, self).__init__()
        self.approach = approach
        self.iterations = iterations
        self.greedy_action_treshold = greedy_action_treshold
        self.epsilon = epsilon

    def plan(self, ei):
        sfl.Planner.lrtdp.LRTDPModule.Lrtdp.clear()
        super(LRTDPPlanner, self).plan(ei)

    def _plan(self, ei):
        return sfl.Planner.lrtdp.LRTDPModule.Lrtdp(ei, epsilon=self.epsilon, iterations=self.iterations,
                                                             greedy_action_treshold=self.greedy_action_treshold, approach=self.approach).lrtdp()

    def get_name(self):
        return "_".join(list(map(str, [self.__class__.__name__, self.approach, self.iterations])))


class EntropyPlanner(AbstractPlanner):
    def __init__(self, threshold = 1.2, batch=1):
        super(EntropyPlanner, self).__init__()
        self.threshold = threshold
        self.batch = batch

    def _plan(self, ei):
        return ei.entropy_next(self.threshold, self.batch)


class ALLTestsPlanner(AbstractPlanner):
    def stop_criteria(self, ei):
        return len(ei.get_optionals_actions()) == 0


class PlannerExperiment(object):
    def __init__(self, planning_file):
        self.planning_file = planning_file
        self.planners = PlannerExperiment.get_planners()

    def experiment(self):
        for planner in self.planners:
            print planner.get_name()
            planner.plan(sfl.Diagnoser.diagnoserUtils.read_json_planning_file(self.planning_file))

    @staticmethod
    def get_planners():
        return [InitialsPlanner(), ALLTestsPlanner(), AbstractPlanner(), HPPlanner(), EntropyPlanner()] + \
               map(lambda x: LRTDPPlanner("entropy", x), range(1, 20)) + map(lambda x: MCTSPlanner("entropy", x), range(1, 20)) + map(lambda x: MCTSPlanner("hp", x), range(1, 20)) + map(lambda x: LRTDPPlanner("hp", x), range(1, 20))
               # map(lambda x: MCTSPlanner("entropy", x * 10), range(1, 20)) + \
               # map(lambda x: MCTSPlanner("hp", x*10), range(1, 20))+ \
               # map(lambda x: LRTDPPlanner("entropy", x * 10), range(1, 20)) + \
               # map(lambda x: LRTDPPlanner("hp", x*10), range(1, 20))

    def get_results(self):
        metrics = {}
        for planner in self.planners:
            metrics[planner.get_name()] = planner.metrics.metrics
        return metrics

    def get_partial_results(self):
        metrics = {}
        for planner in self.planners:
            metrics[planner.get_name()] = planner.partial_metrics
        return metrics


if __name__ == "__main__":
    pe = PlannerExperiment(r"C:\temp\47")
    pe.experiment()
    print pe.get_results()
