__author__ = 'amir'

import csv
import math
import sys

from .Diagnosis import Diagnosis
from .Staccato import Staccato
from .TF import TF

prior_p = 0.05


class Barinel(object):

    def __init__(self):
        self.M_matrix = []
        self.e_vector = []
        self.prior_probs = []
        self.diagnoses = []


    def set_matrix_error(self, M, e):
        self.M_matrix = M
        self.e_vector = e

    def set_prior_probs(self, probs):
        self.prior_probs=probs

    def get_matrix(self):
        return self.M_matrix

    def get_error(self):
        return self.e_vector

    def get_diagnoses(self):
        return self.diagnoses

    def set_diagnoses(self, diagnoses):
        self.diagnoses = diagnoses

    def non_uniform_prior(self, diag):
        comps = diag.get_diag()
        prob = 1
        for i in range(len(comps)):
            prob *= self.prior_probs[comps[i]]
        return prob

    def generate_probs(self):
        new_diagnoses = []
        probs_sum = 0.0
        for diag in self.get_diagnoses():
            dk = 0.0
            if self.prior_probs == []:
                dk = math.pow(prior_p, len(diag.get_diag()))
            else:
                dk = self.non_uniform_prior(diag)
            tf = self.tf_for_diag(diag.get_diag())
            diag.set_probability(tf.maximize() * dk)
            diag.set_from_tf(tf)
            probs_sum += diag.get_prob()
        for diag in self.get_diagnoses():
            if probs_sum < 1e-5:
                # set uniform to avoid nan
                temp_prob = 1.0 / len(self.diagnoses)
            else:
                temp_prob = diag.get_prob() / probs_sum
            diag.set_probability(temp_prob)
            new_diagnoses.append(diag)
        self.set_diagnoses(new_diagnoses)

    def tf_for_diag(self, diagnosis):
        return TF(self.get_matrix(), self.get_error(), diagnosis)

    def run(self):
        self.set_diagnoses([])
        new_diagnoses = []
        diagnoses = Staccato().run(self.get_matrix(), self.get_error())
        for diagnosis in diagnoses:
            new_diagnoses.append(self._new_diagnosis(diagnosis))
        self.set_diagnoses(new_diagnoses)
        self.generate_probs()
        return self.get_diagnoses()

    def _new_diagnosis(self, diagnosis):
        return Diagnosis(diagnosis)
