from .FullMatrixCompSimilarity import FullMatrixCompSimilarity
from .dynamicSpectrum import dynamicSpectrum
from functools import partial


class DynamicSpectrumCompSimilarity(dynamicSpectrum):
    def __init__(self, OriginalScorePercentage):
        super(DynamicSpectrumCompSimilarity, self).__init__()
        self.CompSimilarity = dict()
        self.TestsSimilarity = dict()
        self.ExperimentType = None
        self.OriginalScorePercentage = OriginalScorePercentage
     
    def set_CompSimilarity(self, s):
        self.CompSimilarity = s
    def set_TestsSimilarity(self, s):
        self.TestsSimilarity = s
    def set_ExperimentType(self, t):
        self.ExperimentType = t

    def convertToFullMatrix(self):
        zeros_vector = [0 for _ in self.getprobabilities()]

        def get_test_vector(test, getter=lambda x: 1):
            vector = []
            for c in test:
                vector.extend(zeros_vector[len(vector):c] + [getter(c)])
            vector.extend(zeros_vector[len(vector):])
            return vector

        ans = FullMatrixCompSimilarity(self.OriginalScorePercentage)
        ans.probabilities = list(self.getprobabilities())
        ans.error = list(self.geterror())
        ans.matrix = [get_test_vector(test) for test in self.getTestsComponents()]
        ans.set_CompSimilarity(self.CompSimilarity)
        ans.set_TestsSimilarity(self.TestsSimilarity)
        ans.set_ExperimentType(self.ExperimentType)

        return ans
