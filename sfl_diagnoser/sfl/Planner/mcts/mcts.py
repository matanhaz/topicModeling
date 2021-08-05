"""
A collection of classes and functions for playing certain types of
games. Specifically, an implementation of the MCTS algorithm.
"""
import InstanceNode
import gc
import sys
states = {}


def clear_states():
    global states
    import gc
    states.clear()
    gc.collect()


def generateState(parent, action, ei, approach):
    global states
    key = repr(ei)
    if key not in states:
        state = InstanceNode.InstanceNode(parent, action, ei, approach)
        states[key] = state
    return states[key]


def getStateIfExists(parent, action, ei):
    global states
    key = repr(ei)
    if key not in states:
        return None
    states[key].add_parent(parent, action)
    return states[key]


class ObjectCounter(object):
    def __init__(self):
        gc.collect()
        self.objects = gc.get_objects()
        self.ids = list(map(id, self.objects))

    def count(self):
        from collections import Counter
        gc.collect()
        objects = gc.get_objects()
        new_objs = []
        for obj in objects:
            if id(obj) not in self.ids:
                new_objs.append(obj)
        c = Counter(list(map(lambda x: x.__class__.__name__, filter(lambda x: hasattr(x, "__class__"), new_objs))))
        self.objects = objects
        self.ids = list(map(id, self.objects))


def mcts_uct(ei, iterations, approach):
    """
    Implementation of the UCT variant of the MCTS algorithm.
    """
    clear_states()
    root = generateState(None, None, ei, approach)
    for _ in xrange(iterations):
        uct_iteration(root)
    return root.best_action(c=0)


def uct_iteration(root):
    child = root
    while not child.terminal() and (not child.experiment_instance.AllTestsReached()):
        if not child.fully_expanded():
            child = child.expand()
            break
        else:
            child = child.best_child()
    cost = child.simulation()
    update_parents(child, cost)


def update_parents(child, cost):
    if child is None:
        return
    for action in child.parents:
        parent = child.parents[action]
        parent.visits.setdefault(action, 0)
        parent.value.setdefault(action, 0)
        parent.visits[action] += 1
        parent.value[action] += cost
        update_parents(parent, cost)
