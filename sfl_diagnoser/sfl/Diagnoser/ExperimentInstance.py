import copy
import math
import random
from math import ceil
from .Diagnosis import Diagnosis
from .dynamicSpectrum import dynamicSpectrum
from .Experiment_Data import Experiment_Data
import numpy
from functools import reduce

TERMINAL_PROB = 0.7


class ExperimentInstance(object):
    def __init__(self, initial_tests, error, priors, bugs, pool, components, estimated_pool=None, **kwargs):
        self.initial_tests = initial_tests
        self.error = error
        self.priors = priors
        self.bugs = bugs
        self.pool = pool
        self.components = components
        self.estimated_pool = estimated_pool
        self.reversed_names = dict(map(lambda x: tuple(reversed(x)), self.components.items()))
        # self.experiment_type = experiment_ty
        list(map(lambda attr: setattr(self, attr, kwargs[attr]), kwargs))
        self.diagnoses = []

    def get_component_id(self, component_name):
        return self.reversed_names.get(component_name, None)

    def get_named_bugs(self):
        return list(map(self.components.get, self.bugs))

    def get_id_bugs(self):
        ret_value = list(map(self.get_component_id, self.bugs))
        return ret_value

    def get_experiment_type(self):
        return 'CompSimilarity'

    def initials_to_DS(self):
        ds = self._create_ds()
        ds.TestsComponents = copy.deepcopy([self.pool[test] for test in self.get_initials()])
        ds.probabilities = list(self.priors)
        ds.error=[self.get_error()[test] for test in self.get_initials()]
        ds.tests_names = list(self.get_initials())
        return ds

    def _create_ds(self):
        return dynamicSpectrum()

    def get_optionals_actions(self):
        optionals = [x for x in Experiment_Data().POOL if x not in self.get_initials()]
        return optionals

    def get_components_vectors(self):
        components = {}
        for test in self.initial_tests:
            for component in Experiment_Data().POOL[test]:
                components.setdefault(Experiment_Data().COMPONENTS_NAMES[component], []).append(test)
        return components

    def get_optionals_probabilities(self):
        optionals = self.get_optionals_actions()
        probabilites = [1.0/len(optionals) for _ in optionals]
        return optionals, probabilites

    def get_optionals_probabilities_by_approach(self, approach, *args, **kwargs):
        if approach == "uniform":
            return self.get_optionals_probabilities()
        elif approach == "hp":
            return self.next_tests_by_hp()
        elif approach == "entropy":
            return self.next_tests_by_entropy(*args, **kwargs)
        else:
            raise RuntimeError("approach is not configured")

    def get_components_probabilities(self):
        """
        calculate for each component c the sum of probabilities of the diagnoses that include c
        return dict of (component, probability)
        """
        self.diagnose()
        compsProbs={}
        for d in self.get_diagnoses():
            p = d.get_prob()
            for comp in d.get_diag():
                compsProbs[comp] = compsProbs.get(comp, 0) + p
        return sorted(compsProbs.items(), key=lambda x: x[1], reverse=True)

    def get_components_probabilities_by_name(self):
        return list(map(lambda component: (Experiment_Data().COMPONENTS_NAMES[component[0]], component[1]), self.get_components_probabilities()))

    def next_tests_by_hp(self):
        """
        order tests by probabilities of the components
        return tests and probabilities
        """
        compsProbs = self.get_components_probabilities()
        comps_probabilities = dict(compsProbs)
        optionals = self.get_optionals_actions()
        assert len(optionals) > 0
        tests_probabilities = []
        for test in optionals:
            trace = Experiment_Data().POOL[test]
            test_p = 0.0
            for comp in trace:
                test_p += comps_probabilities.get(comp, 0)
            tests_probabilities.append(test_p)
        if sum(tests_probabilities) == 0.0:
            return self.get_optionals_probabilities()
        tests_probabilities = [abs(x) for x in tests_probabilities]
        tests_probabilities = [x / sum(tests_probabilities) for x in tests_probabilities]
        return optionals, tests_probabilities

    def next_tests_by_prob(self):
        """
        order tests by probabilities of the components
        return tests and probabilities
        """
        compsProbs = self.get_components_probabilities()
        comps_probabilities = dict(compsProbs)
        optionals = self.get_optionals_actions()
        assert len(optionals) > 0
        tests_probabilities = []
        for test in optionals:
            trace = Experiment_Data().ESTIMATED_POOL[test]
            test_p = 0.0
            for comp in trace:
                test_p += comps_probabilities.get(comp, 0) * trace.get(comp, 0)
            tests_probabilities.append(test_p)
        if sum(tests_probabilities) == 0.0:
            return self.get_optionals_probabilities()
        tests_probabilities = [abs(x) for x in tests_probabilities]
        tests_probabilities = [x / sum(tests_probabilities) for x in tests_probabilities]
        return optionals, tests_probabilities

    def next_tests_by_bd(self):
        self.diagnose()
        probabilities = []
        optionals = self.get_optionals_actions()
        for test in optionals:
            p = 0.0
            trace = Experiment_Data().POOL[test]
            for d in self.get_diagnoses():
                p += (d.get_prob() / len(d.get_diag())) * ([x for x in d.get_diag() if x in trace])
            probabilities.append(p)
        probabilities = [abs(x) for x in probabilities]
        probabilities = [x / sum(probabilities) for x in probabilities]
        return optionals, probabilities

    def next_tests_by_entropy(self, threshold = 1.0):
        import sfl.Planner.domain_knowledge as domain_knowledge
        """
        order by InfoGain using entropy
        return tests and probabilities
        """
        probabilities = []
        optionals = []
        threshold_sum = 0.0
        # optionals = self.get_optionals_actions()
        optionals_seperator, tests_probabilities = domain_knowledge.seperator_hp(self)
        # optionals, tests_probabilities = self.next_tests_by_hp()
        for t, p in sorted(list(zip(optionals_seperator, tests_probabilities)), key=lambda x: x[1], reverse=True)[:int(ceil(len(optionals_seperator) * threshold))]:
            # if threshold_sum > threshold:
            #     break
            info = self.info_gain(t)
            if info > 0:
                threshold_sum += p
                probabilities.append(info)
                optionals.append(t)
        if sum(probabilities) == 0.0:
            return self.get_optionals_probabilities()
        probabilities = [x / sum(probabilities) for x in probabilities]
        return optionals, probabilities

    def info_gain(self, test):
        """
        calculate the information gain by test
        """
        fail_test, pass_test = self.next_state_distribution(test)
        ei_fail, p_fail = fail_test
        ei_pass, p_pass = pass_test
        return self.entropy() - (p_fail * ei_fail.entropy() + p_pass * ei_pass.entropy())

    def entropy(self):
        self.diagnose()
        sum = 0.0
        for d in self.get_diagnoses():
            p = d.get_prob()
            sum -= p * math.log(p)
        return sum

    def childs_probs_by_hp(self):
        """
        compute HP for the optionals tests and return dict of (test, prob)
        """
        comps_prob = dict(self.get_components_probabilities()) # tuples of (comp, prob)
        optionals = self.get_optionals_actions()
        assert len(optionals) > 0
        optionals_probs = {}
        for op in optionals:
            trace = Experiment_Data().POOL[op]
            prob = 0
            for comp in trace:
                prob += comps_prob.get(comp, 0)
            optionals_probs[op] = prob
        return optionals_probs

    def bd_next(self):
        optionals, probabilities = self.next_tests_by_bd()
        return numpy.random.choice(optionals, 1, p = probabilities).tolist()[0]

    def hp_next(self):
        optionals, probabilities = self.next_tests_by_hp()
        numpy.random.seed(0)
        return numpy.random.choice(optionals, 1, p = probabilities).tolist()[0]

    def hp_next_by_prob(self):
        optionals, probabilities = self.next_tests_by_prob()
        numpy.random.seed(0)
        return numpy.random.choice(optionals, 1, p=probabilities).tolist()[0]

    def hp_next_by_prob_random(self):
        optionals, probabilities = self.next_tests_by_prob()
        numpy.random.seed(0)
        # return numpy.random.choice(optionals, 1, p=probabilities).tolist()[0]
        return random.choice(optionals)

    def entropy_next(self, threshold = 1.2, batch=1):
        optionals, information =  self.next_tests_by_entropy(threshold)
        return list(map(lambda x: x[0], sorted(list(zip(optionals, information)), reverse=True, key = lambda x: x[1])[:batch]))
        # return numpy.random.choice(optionals, batch, p = information).tolist()

    def random_next(self):
        return random.choice(self.get_optionals_actions())

    def getMaxProb(self):
        self.diagnose()
        maxP=max([x.probability for x in self.get_diagnoses()])
        return maxP

    def isTerminal(self):
        return self.getMaxProb() > TERMINAL_PROB

    def AllTestsReached(self):
        return len(self.get_optionals_actions()) == 0

    def compute_pass_prob(self,action):
        trace = Experiment_Data().POOL[action]
        probs=dict(self.get_components_probabilities())
        pass_Probability = 1.0
        for comp in trace:
            pass_Probability *= 0.999 # probability of 1 fault for each 1000 lines of code
            if comp in probs:
                pass_Probability *= (1-probs[comp]) # add known faults
        return round(pass_Probability, 6)

    def next_state_distribution(self,action):
        pass_Probability = self.compute_pass_prob(action)
        ei_fail = self.simulateTestOutcome(action, 0)
        ei_pass = self.simulateTestOutcome(action, 1)
        return [(ei_fail, pass_Probability), (ei_pass, 1-pass_Probability)]

    def simulate_next_test_outcome(self,action):
        pass_Probability=self.compute_pass_prob(action)
        if random.random() <= pass_Probability:
            return 0
        else:
            return 1

    def simulate_next_ei(self,action):
        outcome = self.simulate_next_test_outcome(action)
        return outcome,self.next_state_distribution(action)[outcome][0]

    def diagnose(self):
        if self.diagnoses == []:
            self.diagnoses = self.initials_to_DS().diagnose()

    def get_named_diagnoses(self):
        self.diagnose()
        named_diagnoses = []
        for diagnosis in self.get_diagnoses():
            named = Diagnosis(list(map(lambda id: Experiment_Data().COMPONENTS_NAMES[id], diagnosis.diagnosis)))
            named.probability = diagnosis.probability
            named_diagnoses.append(named)
        return named_diagnoses

    @staticmethod
    def precision_recall_diag(buggedComps, dg, pr, validComps):
        fp = len([i1 for i1 in dg if i1 in validComps])
        fn = len([i1 for i1 in buggedComps if i1 not in dg])
        tp = len([i1 for i1 in dg if i1 in buggedComps])
        tn = len([i1 for i1 in validComps if i1 not in dg])
        if ((tp + fp) == 0):
            precision = "undef"
        else:
            precision = (tp + 0.0) / float(tp + fp)
            a = precision
            precision = precision * float(pr)
        if ((tp + fn) == 0):
            recall = "undef"
        else:
            recall = (tp + 0.0) / float(tp + fn)
            recall = recall * float(pr)
        return precision, recall

    def calc_precision_recall(self):
        self.diagnose()
        recall_accum=0
        precision_accum=0
        validComps=[x for x in range(max(reduce(list.__add__, Experiment_Data().POOL.values()))) if x not in self.bugs]
        for d in self.get_diagnoses():
            dg=d.diagnosis
            pr=d.probability
            precision, recall = ExperimentInstance.precision_recall_diag(self.bugs, dg, pr, validComps)
            if(recall!="undef"):
                recall_accum=recall_accum+recall
            if(precision!="undef"):
                precision_accum=precision_accum+precision
        return precision_accum,recall_accum

    def calc_wasted_components(self):
        components = list(map(lambda x: x[0], self.get_components_probabilities()))
        wasted = 0.0
        for b in self.bugs:
            if b not in components:
                return float('inf')
            wasted += components.index(b)
        return wasted / len(self.bugs)


    def count_different_cases(self):
        """
        :return: the number of different test cases in the diagnosis
        """
        optional_tests = list(map(lambda enum: enum[1], filter(lambda enum: enum[0] in self.initial_tests, enumerate(Experiment_Data().POOL))))
        return len(set(map(str, optional_tests)))

    def __repr__(self):
        return repr(self.initial_tests)+"-"+repr([name for name,x in self.error.items() if x==1])

    def get_diagnoses(self):
        self.diagnose()
        return self.diagnoses

    def get_initials(self):
        return self.initial_tests

    def get_error(self):
        return self.error

    def simulateTestOutcome(self, next_test, outcome):
        initial_tests = copy.deepcopy(self.initial_tests)
        initial_tests.append(next_test)
        error = dict(self.error)
        error[next_test] = outcome
        return self.create_instance(initial_tests, error)

    def create_instance(self, initial_tests, error):
        return ExperimentInstance(initial_tests, error)

    def addTests(self, next_tests):
        tests_to_add = [next_tests] if type(next_tests) != list else next_tests
        for t in tests_to_add:
            ei = self.simulateTestOutcome(t, self.error[t])
        return ei
