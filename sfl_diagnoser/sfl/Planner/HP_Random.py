__author__ = 'amir'
import sfl.Diagnoser.ExperimentInstance

"""
 basic planners:
 random
 all_tests
 only_initials
 HP
 Entropy

 all functions return tuple of (precision, recall, steps)
"""

def main_HP(ei,status):
    steps = 0
    list_recall = []
    list_pre = []
    ei.diagnose()
    while not (ei.isTerminal() or ei.AllTestsReached() or len(list_recall) > 150):
        if status == 1:
            ei = .sfl.Diagnoser.ExperimentInstance.addTests(ei, ei.hp_next())
        elif status == 2:
            ei = .sfl.Diagnoser.ExperimentInstance.addTests(ei, ei.hp_next_by_prob())
        elif status == 3:
            ei = .sfl.Diagnoser.ExperimentInstance.addTests(ei, ei.random_next())
        steps = steps + 1
        precision, recall = ei.calc_precision_recall()
        list_recall.append(recall)
        list_pre.append(precision)

    return list_pre, list_recall

def main_Random(ei):
    steps = 0
    while not (ei.isTerminal() or ei.AllTestsReached()):
        ei = ei.addTests(ei.random_next())
        steps=steps+1
    precision, recall=ei.calc_precision_recall()
    return precision, recall, steps, repr(ei)

def only_initials(ei):
    steps = 0
    precision, recall=ei.calc_precision_recall()
    return precision, recall, steps, repr(ei)


def all_tests(ei):
    steps = 0
    while not ei.AllTestsReached() :
        ei = ei.addTests(ei.random_next())
        steps=steps+1
    precision, recall=ei.calc_precision_recall()
    return precision, recall, steps, repr(ei)


def main_entropy(ei, *args, **kwargs):
    steps = 0
    while not (ei.isTerminal() or ei.AllTestsReached()):
        ei = ei.addTests(ei.entropy_next(*args, **kwargs))
        steps = steps + 1
    precision, recall=ei.calc_precision_recall()
    return precision, recall, steps, repr(ei)

if __name__=="__main__":
    file = r"C:\projs\lrtdp\instances\40_uniform_8.txt"
    ei= .sfl.Diagnoser.diagnoserUtils.readPlanningFile(file)
    print main_Random(ei)
    print main_HP(ei)
    print main_entropy(ei)
