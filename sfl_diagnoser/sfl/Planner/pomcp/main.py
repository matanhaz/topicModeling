from Planner.pomcp import MCTSPARAMS, EXPERIMENT, DIAGNOSER

from sfl.Planner.pomcp import EXPERIMENTPARAMS

__author__ = 'amir'

from .Diagnoser import diagnoserUtils

def main(ei):
    searchParams= MCTSPARAMS.MCTSPARAMS()#MCTS::PARAMS
    expParams= EXPERIMENTPARAMS.EXPERIMENTPARAMS()# EXPERIMENT::PARAMS

    print "start", ei.calc_precision_recall()
    real = DIAGNOSER.DIAGNOSER(ei, 0.6)#
    simulator = DIAGNOSER.DIAGNOSER(ei.Copy(), 0.6)#

    experiment = EXPERIMENT.EXPERIMENT(real, simulator, expParams, searchParams)#EXPERIMENT
    print "running"
    return experiment.RunMultiple()#

    # return 0#


if __name__=="__main__":
    file="C:\projs\ptry\lrtdp\\10_0.6_0.0_15_all.txt"
    file="C:\projs\ptry\lrtdp\\10_uniform_15.txt"
    file="C:\projs\ptry\lrtdp\\10_uniform_15_all.txt"
    file="C:\projs\ptry\lrtdp\\10_0.6_0.0_15.txt"
    ei= .sfl.Diagnoser.diagnoserUtils.readPlanningFile(file)