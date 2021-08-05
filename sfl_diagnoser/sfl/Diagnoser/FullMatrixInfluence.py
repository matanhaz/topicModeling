from .BarinelInfluence import BarinelInfluence
from .FullMatrix import FullMatrix


class FullMatrixInfluence(FullMatrix):
    def __init__(self):
        super(FullMatrixInfluence, self).__init__()
        self.influence_matrix = dict()
        self.influence_alpha = 0

    def set_influence_matrix(self, matrix):
        self.influence_matrix = matrix

    def set_influence_alpha(self, alpha):
        self.influence_alpha = alpha

    def diagnose(self):
        bar = BarinelInfluence()
        bar.set_matrix_error(self.matrix,self.error)
        bar.set_prior_probs(self.probabilities)
        bar.set_influence_matrix(self.influence_matrix)
        bar.set_influence_alpha(self.influence_alpha)
        return bar.run()


    # optimization: remove unreachable components & components that pass all their tests
    # return: optimized FullMatrix, chosen_components( indices), used_tests
    def optimize(self):
        optimizedMatrix, used_components, used_tests = super(FullMatrixInfluence, self).optimize()
        new_influence_matrix = FullMatrixInfluence()
        new_influence_matrix.set_error(optimizedMatrix.error)
        new_influence_matrix.set_matrix(optimizedMatrix.matrix)
        new_influence_matrix.set_probabilities(optimizedMatrix.probabilities)
        new_matrix = []
        for test in used_tests:
            new_test = []
            for i in range(len(used_components)):
                new_test.append(self.influence_matrix[test][used_components[i]])
            new_matrix.append(new_test)
        new_influence_matrix.set_influence_matrix(new_matrix)
        new_influence_matrix.set_influence_alpha(self.influence_alpha)
        return new_influence_matrix, used_components, used_tests
