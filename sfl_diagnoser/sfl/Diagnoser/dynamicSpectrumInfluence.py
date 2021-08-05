from .FullMatrixInfluence import FullMatrixInfluence
from .dynamicSpectrum import dynamicSpectrum
from functools import partial


class DynamicSpectrumInfluence(dynamicSpectrum):
    def __init__(self):
        super(DynamicSpectrumInfluence, self).__init__()
        self.influence_matrix = dict()
        self.influence_alpha = 0

    def set_influence_matrix(self, matrix):
        self.influence_matrix = matrix

    def set_influence_alpha(self, alpha):
        self.influence_alpha = alpha

    def convertToFullMatrix(self):
        zeros_vector = [0 for _ in self.getprobabilities()]

        def get_test_vector(test, getter=lambda x: 1):
            vector = []
            for c in test:
                vector.extend(zeros_vector[len(vector):c] + [getter(c)])
            vector.extend(zeros_vector[len(vector):])
            return vector

        ans = FullMatrixInfluence()
        ans.probabilities = list(self.getprobabilities())
        ans.error = list(self.geterror())
        ans.matrix = [get_test_vector(test) for test in self.getTestsComponents()]
        influence_matrix = []
        for test_id, test in enumerate(self.getTestsComponents()):
            influence_matrix.append(get_test_vector(test, self.influence_matrix[test_id].get))
        ans.influence_matrix = influence_matrix
        ans.influence_alpha = self.influence_alpha
        return ans
