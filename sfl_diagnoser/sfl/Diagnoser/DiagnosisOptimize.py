__author__ = 'amir'

from.Diagnosis import Diagnosis


class DiagnosisOptimize(Diagnosis):
    def __init__(self, diagnosis=None):
        super(DiagnosisOptimize, self).__init__(diagnosis=diagnosis)
        self.key = ""

    def clone(self):
        res = DiagnosisOptimize()
        res.diagnosis = list(self.diagnosis)
        res.probability = self.get_prob()
        return res

    def set_from_tf(self, tf):
        self.key = tf.key

    def __repr__(self):
        return str(sorted(self.diagnosis)) + " P: " + str(self.get_prob()) + " key: " + self.key
