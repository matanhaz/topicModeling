from .Barinel import Barinel
from functools import reduce


class FullMatrix(object):
    def __init__(self):
        self.matrix=[] # matrix size define the num of tests
        self.probabilities=[] # probabilities size define the num of components
        self.error=[]

    def diagnose(self):
        bar = self._create_barinel()
        bar.set_matrix_error(self.matrix,self.error)
        bar.set_prior_probs(self.probabilities)
        return bar.run()

    def _create_barinel(self):
        return Barinel()

    def save_to_csv_file(self, out_file):
        import csv
        lines = [self.probabilities] + list(map(lambda x: x[0] + [x[1]], zip(self.matrix, self.error)))
        with open(out_file, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(lines)

    def set_matrix(self, matrix):
        self.matrix = matrix

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    def set_error(self, error):
        self.error = error

    # optimization: remove unreachable components & components that pass all their tests
    # return: optimized FullMatrix, chosen_components( indices), used_tests
    def optimize(self):
        failed_tests = list(map(lambda test: list(enumerate(test[0])), filter(lambda test: test[1] == 1, zip(self.matrix, self.error))))
        used_components = dict(enumerate(sorted(reduce(set.__or__, map(lambda test: set(map(lambda comp: comp[0], filter(lambda comp: comp[1] == 1, test))), failed_tests), set()))))
        optimizedMatrix = self._create_full_matirx()
        optimizedMatrix.set_probabilities([x[1] for x in enumerate(self.probabilities) if x[0] in used_components])
        newErr = []
        newMatrix = []
        used_tests = []
        for i, (test, err) in enumerate(zip(self.matrix, self.error)):
            newTest = list(map(lambda i: test[i], sorted(used_components.values())))
            if 1 in newTest: ## optimization could remove all comps of a test
                newMatrix.append(newTest)
                newErr.append(err)
                used_tests.append(i)
        optimizedMatrix.set_matrix(newMatrix)
        optimizedMatrix.set_error(newErr)
        return optimizedMatrix, used_components, sorted(used_tests)

    def _create_full_matirx(self):
        return FullMatrix()
