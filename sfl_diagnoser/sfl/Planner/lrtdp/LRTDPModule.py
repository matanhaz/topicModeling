__author__ = 'amir'

import time

import sfl.Diagnoser.ExperimentInstance
import sfl.Planner.lrtdp.lrtdpState


class Lrtdp(object):
    states = dict()

    def __init__(self, experiment_instance, epsilon=0.001, iterations=1, greedy_action_treshold=1, approach="uniform"):
        self.epsilon = epsilon
        self.iterations = iterations
        self.greedy_action_treshold = greedy_action_treshold
        self.experiment_instance = experiment_instance
        self.approach = approach

    def lrtdp(self):
        state = self.create_start_state()
        for _ in xrange(self.iterations):
            if state.isSolved:
                break
            self.runLrtdpTrial(state)
        return state.greedyAction(self.greedy_action_treshold)

    def create_start_state(self):
        return self.generateState(self.experiment_instance)

    def generateState(self, ei):
        key = repr(ei)
        if key not in Lrtdp.states:
            state = sfl.Planner.lrtdp.lrtdpState.LrtdpState(ei, self.approach, self)
            Lrtdp.states[key] = state
        return Lrtdp.states[key]

    @staticmethod
    def clear():
        Lrtdp.states.clear()
        Lrtdp.states = dict()

    def nextStateDist(self, ei, action):
        dist = ei.next_state_distribution(action)
        stateDist = []
        for next, prob in dist:
            stateDist.append((self.generateState(next), prob))
        return stateDist

    def runLrtdpTrial(self, state):
        visited = []  # stack
        while not (state.isSolved or state.AllTestsReached()):
            visited.append(state)
            if state.isTerminal():
                break
            a = state.greedyAction(self.greedy_action_treshold)
            state.update(a)
            state = state.simulate_next_state(a)
        while visited:
            if not self.check_solved(visited.pop()):
                break

    def check_solved(self, state):
        rv = True
        open = []
        closed = {}
        if not state.isSolved:
            open.append(state)
        while open:
            state = open.pop()
            if state.AllTestsReached():
                continue
            a = state.greedyAction()
            closed[state] = a
            if state.residual(a) > self.epsilon:
                rv = False
                break
            for next, prob in state.getNextStateDist(a):
                if (not next.isSolved) and (next not in open) and (next not in closed):
                    open.append(next)
        if rv:
            for c in closed:
                c.isSolved = True
        else:
            for c in closed:
                if not c.AllTestsReached():
                    c.update(closed[c])
        return rv

    def evaluatePolicy(self):
        state = self.create_start_state()
        steps = 0
        ei = state.experimentInstance
        while (not state.isSolved) and (not state.terminal_or_allReach()):
            action = state.greedyAction()
            ei = ei.addTests(action)
            state = self.generateState(ei)
            steps += 1
        precision, recall = ei.calc_precision_recall()
        return precision, recall, steps, repr(ei)

    def multiLrtdp(self):
        state = self.create_start_state()
        trialsCount = 0
        steps = 0
        ei = state.experimentInstance
        if state.isTerminal():
            precision, recall = ei.calc_precision_recall()
            return precision, recall, 0
        while not state.isSolved:
            if trialsCount > self.iterations:
                return
            trialsCount += 1
            success = self.runLrtdpTrial(state)
            if not success:
                return
            a = state.greedyAction()
            ei = ei.addTests(a)
            state = self.generateState(ei)
            steps += 1
        precision, recall = ei.calc_precision_recall()
        return precision, recall, steps
