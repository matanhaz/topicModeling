__author__ = 'amir'

#from pyswarm import pso
#from LightPSO import LightPSO
import operator
import functools
from collections import Counter
from functools import reduce


class TF(object):
    def __init__(self, matrix, error, diagnosis):
        self.activity = list(zip(range(len(matrix)), map(tuple, matrix), error))
        self.diagnosis = diagnosis
        self.active_components = dict(map(lambda a: (a[0], list(filter(functools.partial(tuple.__getitem__, a[1]), self.diagnosis))), self.activity))
        self.max_value = None

    def get_active_components(self):
        return self.active_components

    def set_max_value(self, value=None):
        self.max_value = value

    def get_max_value(self):
        return self.max_value

    def get_activity(self):
        return self.activity

    def get_diagnosis(self):
        return self.diagnosis

    def probabilty(self, h_dict):
        # h_dict is dict of dicts for test to comps
        def test_prob(test_id, v, e):
            # if e==0 : h1*h2*h3..., if e==1: 1-h1*h2*h3...
            return e + ((-2.0 * e + 1.0) * reduce(operator.mul,
                                                   list(map(h_dict[test_id].get, self.get_active_components()[test_id])), 1.0))
        x = reduce(operator.mul, list(map(lambda x: test_prob(*x), self.get_activity())), 1.0)
        return x

    def probabilty_TF(self, h):
        dict_test = dict(zip(self.get_diagnosis(), h))
        return -self.probabilty(dict(map(lambda i: (i, dict_test), range(len(self.get_activity())))))

    def not_saved(self):
        pass

    def maximize(self):
        if self.max_value is None:
            self.not_saved()
            initialGuess=[0.1 for _ in self.get_diagnosis()]
            lb = [0 for _ in self.get_diagnosis()]
            ub = [1 for _ in self.get_diagnosis()]
            import scipy.optimize
            self.max_value = -scipy.optimize.minimize(self.probabilty_TF,initialGuess,method="L-BFGS-B"
                                        ,bounds=list(zip(lb,ub)), tol=1e-3,options={'maxiter':100}).fun
            # self.max_value = -scipy.optimize.minimize(self.probabilty_TF,initialGuess,method="TNC"
            #                             ,bounds=zip(lb,ub), tol=1e-2,options={'maxiter':10}).fun
            # self.max_value = -scipy.optimize.minimize(self.probabilty_TF,initialGuess,method="SLSQP"
            #                             ,bounds=zip(lb,ub), tol=1e-2,options={'maxiter':10}).fun
            # self.max_value = -scipy.optimize.minimize(self.probabilty_TF,initialGuess,method="trust-constr"
            #                             ,bounds=zip(lb,ub), tol=1e-2,options={'maxiter':10}).fun
            # self.max_value = self.maximize_by_gradient()
            # self.max_value = -pso(self.probabilty_TF, lb, ub, minfunc=1e-3, minstep=1e-3, swarmsize=20,maxiter=10)[1]
            # print "size", len(initialGuess)
            # self.max_value = -self.probabilty_TF(initialGuess)
            # self.max_value = -LightPSO(len(self.diagnosis), self).run()
        return self.max_value

    def calculate(self, values):
        return self.probabilty(values)

    def maximize_by_gradient(self):
        gamma = 0.01
        precision = 1e-3
        prOld = 0
        pr = 1.0
        i = 0
        gj = dict((comp, 0.1) for comp in self.self.get_diagnosis())
        while abs(prOld - pr) > precision:
            i += 1
            prOld = pr
            gradients = self.gradient(gj)
            for comp in self.self.get_diagnosis():
                val = gj[comp] + gamma * gradients[comp]
                gj[comp] = max(min(val, 1), 0)
            pr = self.calculate(gj)
        return pr

    def gradient(self, vals):
        margin = 0.1
        new_vals = {}
        for comp in vals:
            d1 = self.centralDividedDifference(vals, comp, margin)
            d2 = self.centralDividedDifference(vals, comp, margin/2)
            new_vals[comp] = d2 + ((d2 - d1) / 3)
        return new_vals

    def centralDividedDifference(self, vals, comp, margin):
        val = vals[comp]
        vals[comp] = val + margin
        plus = self.calculate(vals)
        vals[comp] = val - margin
        minus = self.calculate(vals)
        vals[comp] = val
        return (plus-minus)/(2*margin)
