__author__ = 'amir'
from math import sqrt, log

import sfl.Planner.mcts.actionNode
import numpy


class ChildActions(object):
    def __init__(self, approach, actions, probabilities, instance_node):
        self.instance_node = instance_node
        self.approach = approach
        self.children = dict()
        self.actions = sorted(zip(actions, probabilities), key=lambda x: x[1], reverse=True)

    def __getitem__(self, action):
        if action not in self.children:
            self.children[action] = sfl.Planner.mcts.actionNode.ActionNode(action, self.instance_node, self.approach)
        return self.children[action]

    def fully_expanded(self):
        return len(self.actions) == len(self.children) and all(list(map(lambda x: x.fully_expanded(), self.children.values())))

    def get_children(self, all_children=True):
        if all_children:
            return list(zip(*self.actions))[0]
        return self.children.keys()


class InstanceNode(object):
    def __init__(self, parent, action, experiment_instance, approach):
        """
        approach - how to combine tests probabilities to qvalue.
            can be one of the following: "uniform" , "hp", "entropy"
        """
        self.experiment_instance = experiment_instance
        self.approach = approach
        self.parents = dict()
        if not parent is None:
            self.parents[action] = parent
        actions, probabilities = self.experiment_instance.get_optionals_probabilities_by_approach(self.approach)
        self.children_ = ChildActions(self.approach, actions, probabilities, self)
        self.visits = dict()
        self.value = dict()

    @property
    def weight(self):
        """
        The weight of the current node.
        """
        if sum(self.visits.values()) == 0:
            return 0
        return sum(self.value.values()) / float(sum(self.visits.values()))

    def search_weight(self, c):
        """
        Compute the UCT search weight function for this node. Defined as:

            w = Q(v') / N(v') + c * sqrt(2 * ln(N(v)) / N(v'))

        Where v' is the current node and v is the parent of the current node,
        and Q(x) is the total value of node x and N(x) is the number of visits
        to node x.
        """
        weight = self.weight
        for p in self.parents:
            parent = self.parents[p]
            if sum(parent.visits.values()) == 0:
                continue
            weight += c * sqrt(2 * log(sum(parent.visits.values())) / sum(self.visits.values()))
        return weight

    def add_parent(self, parent, action):
        self.parents[action] = parent

    def result(self, action):
        """
        The state resulting from the given action taken on the current node
        state by the node player.
        """
        return self.experiment_instance.simulate_next_ei(action)[1]

    def terminal(self):
        """
        Whether the current node state is terminal.
        """
        return self.experiment_instance.isTerminal()

    def fully_expanded(self):
        """
        Whether all child nodes have been expanded (instantiated). Essentially
        this just checks to see if any of its children are set to None.
        """
        return self.children_.fully_expanded()

    def expand(self):
        """
        Instantiates one of the unexpanded children (if there are any,
        otherwise raises an exception) and returns it.
        """
        try:
            for action in self.children_.get_children():
                if not self.children_[action].fully_expanded():
                    return self.children_[action].expand()
        except ValueError:
            raise Exception('Node is already fully expanded')

    def best_child(self, c=1/sqrt(2)):
        """
        return the action with max search_weight + hp. in case that action not expanded weight = 0 .
        """
        # if not self.fully_expanded():
        #     raise Exception('Node is not fully expanded')
        values = []
        for action in self.children_.get_children():
            weight = self.children_[action].search_weight(c)
            values.append((action, weight))
        action = max(values, key=lambda x: x[1])[0]
        return self.children_[action].expand()

    def best_action(self, c=1/sqrt(2)):
        """
        Returns the action needed to reach the best child from the current
        node.
        """
        child = self.best_child(c)
        action = list(filter(lambda parent: parent[1] == self, child.parents.items()))[0][0]
        return action, child.weight

    def simulation(self):
        """
        Simulates the game to completion, choosing moves in a uniformly random
        manner. The outcome of the simulation is returns as the state value for
        the given player.
        """
        steps = 1
        ei = self.experiment_instance
        while (not ei.isTerminal()) and (not ei.AllTestsReached()):
            optionals, probabilities = self.experiment_instance.get_optionals_probabilities_by_approach(self.approach)
            action = numpy.random.choice(optionals, p=probabilities)
            ei = ei.simulate_next_ei(action)[1]
            steps += 1
        if not ei.isTerminal():
            steps = float('inf')
        return -steps
