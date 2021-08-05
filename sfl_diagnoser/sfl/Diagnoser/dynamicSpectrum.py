from .FullMatrix import FullMatrix
from functools import reduce


class dynamicSpectrum(object):
    def __init__(self):
        self.TestsComponents = [] # TestsComponents size define the num of tests
        self.probabilities = [] # probabilities size define the num of components
        self.tests_names = []
        self.error = []

    def getTestsComponents(self):
        return self.TestsComponents

    def getprobabilities(self):
        return self.probabilities

    def gettests_names(self):
        return self.tests_names

    def geterror(self):
        return self.error

    def setTestsComponents(self, TestsComponents):
        self.TestsComponents = TestsComponents

    def setprobabilities(self, probabilities):
        self.probabilities = probabilities

    def settests_names(self, tests_names):
        self.tests_names = tests_names

    def seterror(self, error):
        self.error = error

    def convertToFullMatrix(self):
        ans = self._create_full_matrix()
        ans.set_probabilities(list(self.getprobabilities()))
        ans.set_error(list(self.geterror()))
        ans.set_matrix(list(map(lambda test: list(map(lambda comp: 1 if comp in test else 0, range(len(self.getprobabilities())))), self.getTestsComponents())))
        return ans

    def _create_full_matrix(self):
        return FullMatrix()

    def remove_duplicate_tests(self):
        distinct_tests = set(map(str, zip(self.getTestsComponents(), self.geterror())))
        names = []
        chosen = []
        error = []
        for name, test, e in zip(self.gettests_names(), self.getTestsComponents(), self.geterror()):
            repr = str((test, e))
            if repr in distinct_tests:
                names.append(name)
                chosen.append(test)
                error.append(e)
                distinct_tests.remove(repr)
        ds = dynamicSpectrum()
        ds.probabilities = self.getprobabilities()
        ds.tests_names= names
        ds.TestsComponents = chosen
        ds.error = error
        return ds

    def get_tests_by_error(self, error):
        return list(map(lambda test: test[1], filter(lambda test: self.geterror()[test[0]] == error, enumerate(self.getTestsComponents()))))

    def get_failed_tests(self):
        return self.get_tests_by_error(1)

    def get_passed_tests(self):
        return self.get_tests_by_error(0)

    def get_components(self):
        return set(reduce(list.__add__, self.getTestsComponents()))

    def get_components_in_failed_tests(self):
        return set(reduce(list.__add__, self.get_failed_tests(), []))

    def optimize(self):
        """
        return new DS object without components that not included in any failed tests.
        """
        failed_components = self.get_components_in_failed_tests()
        new_names = []
        new_tests = []
        new_probabilities = list(map(lambda p: p[1], filter(lambda p: p[0] in failed_components, enumerate(self.getprobabilities()))))
        new_error = []
        for name, test, e in zip(self.gettests_names(), self.getTestsComponents(), self.geterror()):
            new_test = list(filter(lambda comp: comp in failed_components, test))
            if new_test == []:
                continue
            new_names.append(name)
            new_tests.append(new_test)
            new_error.append(e)
        ds = dynamicSpectrum()
        ds.probabilities = new_probabilities
        ds.tests_names = new_names
        ds.TestsComponents = new_tests
        ds.error = new_error
        return ds

    #return diagnoses
    def diagnose(self):
        fullM, used_components, used_tests = self.convertToFullMatrix().optimize()
        Opt_diagnoses = fullM.diagnose()
        diagnoses = []
        for diag in Opt_diagnoses:
            diag=diag.clone()
            diag_comps=[used_components[x] for x in diag.diagnosis]
            diag.diagnosis=list(diag_comps)
            diagnoses.append(diag)
        return diagnoses
