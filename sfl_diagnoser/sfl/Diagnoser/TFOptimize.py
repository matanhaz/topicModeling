__author__ = 'amir'

import operator
import functools
from .TF import TF
import json
import os
from collections import Counter
import atexit

MEMOIZE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), r"TFOptimize_memoize.json"))


class TfMemoize(object):
    def __init__(self):
        self.memo = {}
        if os.path.exists(MEMOIZE_PATH):
            with open(MEMOIZE_PATH) as f:
                print(MEMOIZE_PATH)
                self.memo = json.load(f)
        atexit.register(self.save)

    def save(self):
        with open(MEMOIZE_PATH, "wb") as f:
            json.dump(self.memo, f)

    def memoize(self, f):
        def helper(tf):
            if tf.key not in self.memo:
                self.memo[tf.key] = f(tf)
            return self.memo[tf.key]
        return helper


#memoize = TfMemoize()


class TFOptimize(TF):
    def __init__(self, matrix, error, diagnosis):
        super(TFOptimize, self).__init__(matrix, error, diagnosis)
        self.key = TFOptimize.get_key(matrix, error, diagnosis)

    @staticmethod
    def get_key(matrix, error, diagnosis):
        s_passed = []
        s_failed = []
        for line, e in zip(matrix, error):
            new_line = tuple(map(line.__getitem__, diagnosis))
            sum1 = sum(new_line)
            if sum1:
                if e:
                    s_failed.append(sum1)
                else:
                    s_passed.append(sum1)
        range1 = range(1, 1 + len(diagnosis))
        return ".".join(["-".join(list(map(lambda x: str(Counter(s_passed).get(x, 0)), range1))),
                         "-".join(list(map(lambda x: str(Counter(s_failed).get(x, 0)), range1)))])

    #@memoize.memoize
    def maximize(self):
        return super(TFOptimize, self).maximize()
