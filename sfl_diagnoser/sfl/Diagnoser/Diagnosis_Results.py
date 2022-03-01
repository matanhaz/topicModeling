from .Experiment_Data import Experiment_Data
from .Ochiai_Rank import Ochiai_Rank
from scipy.stats import entropy
from collections import Counter
import numpy as np
from functools import reduce


class Diagnosis_Results(object):
    def __init__(self, diagnoses, initial_tests, error, pool=None, bugs=None):
        self.diagnoses = diagnoses
        self.initial_tests = initial_tests
        self.error = error
        self.pool = pool
        self.bugs = bugs
        if bugs is None:
            experiment_data_bugs = bugs
            if isinstance(experiment_data_bugs[0], int):
                self.bugs = experiment_data_bugs
            else:
                self.bugs = Experiment_Data().get_id_bugs()
        self.components = set(reduce(list.__add__, list(map(self.pool.get, self.initial_tests)), []))
        self.metrics = self._calculate_metrics()
        for key, value in self.metrics.items():
            setattr(self, key, value)

    @staticmethod
    def diagnosis_results_from_experiment_instance(experiment_instance):
        return Diagnosis_Results(experiment_instance.diagnoses,
                                 experiment_instance.initial_tests,
                                 experiment_instance.error)

    def _calculate_metrics(self):
        """
        calc result for the given experiment instance
        :param experiment_instance:
        :return: dictionary of (metric_name, metric value)
        """
        metrics = {}
        precision, recall = self.calc_precision_recall()
        metrics["fscore"] = (precision * recall * 2) / (precision + recall) if (precision + recall) != 0 else 0
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["entropy"] = self.calc_entropy()
        metrics["component_entropy"] = self.calc_component_entropy()
        metrics["num_comps"] = len(self.get_components())
        metrics["num_diagnoses"] = len(self.diagnoses)
        metrics["distinct_diagnoses_scores"] = len(Counter(list(map(lambda x: x.probability, self.diagnoses))))
        metrics["num_tests"] = len(self.get_tests())
        metrics["num_distinct_traces"] = len(self.get_distinct_traces())
        metrics["num_failed_tests"] = len(self._get_tests_by_error(1))
        metrics["num_passed_tests"] = len(self._get_tests_by_error(0))
        passed_comps = set(self._get_components_by_error(0))
        failed_comps = set(self.get_components_in_failed_tests())
        metrics["num_failed_comps"] = len(failed_comps)
        metrics["only_failed_comps"] = len(failed_comps - passed_comps)
        metrics["only_passed_comps"] = len(passed_comps - failed_comps)
        metrics["num_bugs"] = len(self.get_bugs())
        metrics["wasted"] = self.calc_wasted_components()
        metrics["top_k"] = self.calc_top_k()
        metrics["num_comps_in_diagnoses"] = len(self._get_comps_in_diagnoses())
        metrics["bugs_cover_ratio"] = self._get_bugs_cover_ratio()
        metrics["average_trace_size"] = self._get_average_trace_size()
        metrics["average_component_activity"] = self._get_average_component_activity()
        metrics["average_diagnosis_size"] = self._get_average_diagnosis_size()
        metrics["bugs_scores_average"], metrics["bugs_scores_std"], metrics["bugs_scores_entropy"] = self._get_bugs_scores()
        metrics["non_bugs_scores_average"], metrics["non_bugs_scores_std"], metrics["non_bugs_scores_entropy"] = self._get_non_bugs_scores()
        metrics.update(self.cardinality())
        # metrics["ochiai"] = self.calc_ochiai_values()
        return metrics

    def _get_comps_in_diagnoses(self):
        return reduce(set.__or__, list(map(lambda x: set(x.diagnosis), self.diagnoses)), set())

    def _get_bugs_cover_ratio(self):
        bugs = self.get_bugs()
        comps = self._get_comps_in_diagnoses()
        return len(set(comps) & set(bugs)) / (len(bugs) * 1.0)

    def _get_bugs_scores(self):
        bugs = self.get_bugs()
        comps_prob = dict(self.get_components_probabilities())
        bugs_prob = list(map(lambda x: comps_prob.get(x, 0), bugs))
        return np.mean(bugs_prob), np.std(bugs_prob), entropy(bugs_prob)

    def _get_average_trace_size(self):
        return np.mean(list(map(len, self.pool.values())))

    def _get_average_diagnosis_size(self):
        return np.mean(list(map(lambda x: len(x.diagnosis), self.diagnoses)))

    def _get_average_component_activity(self):
        return np.mean(list(Counter(reduce(list.__add__, self.pool.values(), [])).values()))

    def _get_non_bugs_scores(self):
        bugs = self.get_bugs()
        comps_prob = dict(self.get_components_probabilities())
        non_bugs_prob = list(map(comps_prob.get, filter(lambda c: c not in bugs, comps_prob)))
        return np.mean(non_bugs_prob), np.std(non_bugs_prob), entropy(non_bugs_prob)

    def _get_metrics_list(self):
        return sorted(self.metrics.items(), key=lambda m:m[0])

    def get_metrics_values(self):
        return list(map(lambda m:m[1], self._get_metrics_list()))

    def get_metrics_names(self):
        return list(map(lambda m:m[0], self._get_metrics_list()))

    def __repr__(self):
        return repr(self.metrics)

    @staticmethod
    def precision_recall_for_diagnosis(buggedComps, dg, pr, validComps):
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
        recall_accum=0
        precision_accum=0
        validComps=[x for x in set(reduce(list.__add__, self.pool.values())) if x not in self.get_bugs()]

        self.diagnoses.sort(key = lambda x: x.probability, reverse = True)

        self.diagnoses = self.diagnoses[:5]

        for d in self.diagnoses:
            dg=d.diagnosis
            pr=d.probability
            precision, recall = Diagnosis_Results.precision_recall_for_diagnosis(self.get_bugs(), dg, pr, validComps)
            if(recall!="undef"):
                recall_accum=recall_accum+recall
            if(precision!="undef"):
                precision_accum=precision_accum+precision
        return precision_accum,recall_accum

    def get_tests(self):
        return self.pool.items()

    def get_bugs(self):
        return self.bugs

    def get_initial_tests_traces(self):
        return list(map(lambda test: (sorted(self.pool[test]), self.error[test]), self.initial_tests))

    def _get_tests_by_error(self, error):
        by_error = list(filter(lambda test: self.error[test] == error, self.initial_tests))
        return dict(map(lambda test: (test, self.pool[test]), by_error))

    def get_components(self):
        return set(reduce(list.__add__, self.pool.values()))

    def _get_components_by_error(self, error):
        return set(reduce(list.__add__, self._get_tests_by_error(error).values(), []))

    def get_components_in_failed_tests(self):
        return self._get_components_by_error(1)

    def get_components_in_passed_tests(self):
        return self._get_components_by_error(0)

    def get_components_probabilities(self):
        """
        calculate for each component c the sum of probabilities of the diagnoses that include c
        return dict of (component, probability)
        """
        compsProbs={}
        for d in self.diagnoses:
            p = d.get_prob()
            for comp in d.get_diag():
                compsProbs[comp] = compsProbs.get(comp,0) + p
        return sorted(compsProbs.items(), key=lambda x: x[1], reverse=True)

    def calc_wasted_components(self):
        components = list(map(lambda x: x[0], self.get_components_probabilities()))
        if len(self.get_bugs()) == 0:
            return len(components)
        wasted = 0.0
        for b in self.get_bugs():
            if b not in components:
                return len(components)
            wasted += components.index(b)
        return wasted / len(self.get_bugs())

    def calc_top_k(self):
        components = list(map(lambda x: x[0], self.get_components_probabilities()))
        top_k = None
        for bug in self.get_bugs():
            if bug in components:
                if top_k:
                    top_k = max(top_k, components.index(bug))
                else:
                    top_k = components.index(bug)
        return top_k or len(components)

    def calc_entropy(self):
        return entropy(list(map(lambda diag: diag.probability, self.diagnoses)))

    def calc_component_entropy(self):
        return entropy(list(map(lambda x: x[1], self.get_components_probabilities())))

    def get_uniform_entropy(self):
        uniform_probability = 1.0/len(self.diagnoses)
        return entropy(list(map(lambda diag: uniform_probability, self.diagnoses)))

    def get_distinct_traces(self):
        distinct_tests = set(map(str, self.get_initial_tests_traces()))
        return distinct_tests

    def calc_ochiai_values(self):
        ochiai = {}
        for component in self.components:
            ochiai[component] = Ochiai_Rank()
        for trace, error in self.get_initial_tests_traces():
            for component in self.components:
                    ochiai[component].advance_counter(1 if component in trace else 0, error)
        return ochiai

    def cardinality(self):
        import pandas as pd
        d = pd.DataFrame(list(map(lambda d: len(d.diagnosis), self.diagnoses))).describe()
        return dict(map(lambda x: ("cardinality" + x, d.loc[x][0]), d.index.values))
