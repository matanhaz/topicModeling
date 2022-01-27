from .BarinelCompSimilarity import BarinelCompSimilarity
from .FullMatrix import FullMatrix


class FullMatrixCompSimilarity(FullMatrix):
    def __init__(self, OriginalScorePercentage):
        super(FullMatrixCompSimilarity, self).__init__()
        self.CompSimilarity = {}
        self.TestsSimilarity = dict()
        self.ExperimentType = None
        self.OriginalScorePercentage = OriginalScorePercentage
     
    def set_CompSimilarity(self, s):
        self.CompSimilarity = s
    def set_TestsSimilarity(self, s):
        self.TestsSimilarity = s
    def set_ExperimentType(self, t):
        self.ExperimentType = t
   

    def diagnose(self):
        bar = BarinelCompSimilarity(self.OriginalScorePercentage)
        bar.set_matrix_error(self.matrix,self.error)
        bar.set_prior_probs(self.probabilities)
        bar.set_CompSimilarity(self.CompSimilarity)
        bar.set_TestsSimilarity(self.TestsSimilarity)
        bar.set_ExperimentType(self.ExperimentType)
        return bar.run()


    # optimization: remove unreachable components & components that pass all their tests
    # return: optimized FullMatrix, chosen_components( indices), used_tests
    def optimize(self):
        optimizedMatrix, used_components, used_tests = super(FullMatrixCompSimilarity, self).optimize()
        new_CompSimilarity_matrix = FullMatrixCompSimilarity(self.OriginalScorePercentage)
        new_CompSimilarity_matrix.set_error(optimizedMatrix.error)
        new_CompSimilarity_matrix.set_matrix(optimizedMatrix.matrix)
        new_CompSimilarity_matrix.set_probabilities(optimizedMatrix.probabilities)
        CompSimilarity = {}
        for i in range(len(used_components)):
            CompSimilarity[i] = self.CompSimilarity[used_components[i]]
        TestsSimilarity = {}
        for i in range(len(used_tests)):
            TestsSimilarity[i] = self.TestsSimilarity[used_tests[i]]
        new_CompSimilarity_matrix.set_CompSimilarity(CompSimilarity)
        new_CompSimilarity_matrix.set_TestsSimilarity(TestsSimilarity)
        new_CompSimilarity_matrix.set_ExperimentType(self.ExperimentType)

        return new_CompSimilarity_matrix, used_components, used_tests
