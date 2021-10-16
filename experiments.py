
from os.path import join, exists
from os import mkdir, listdir

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm


class Experiment_1:
    def __init__(self, project_name):
        self.experiment_path = join('projects', project_name, 'Experiments', 'Experiment_1')
        self.data_path = join('projects', project_name, 'Experiments', 'Experiment_1', 'data')
        self.results_path = join('projects', project_name, 'Experiments', 'results')
        self.x = []
        self.y = []

    def __call__(self):
        for file in listdir(self.data_path):
            if 'topicModeling' in file:
                self.run(file)
            else:
                self.run(file, 3, 4)

        self.save_plot(self.x, self.y, 'num topics / method', 'average max index of all functions in results', 'Experiment_1_results')

    def save_plot(self, x, y, x_label, y_label, file_name):
        if not exists(self.results_path):
            mkdir(self.results_path)
        plt.figure(figsize=(15, 6))
        plt.bar(x,y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Experiment 1')
        plt.savefig(join(self.results_path, file_name), dpi=100)
        #plt.show()

    def run(self, file_name='topicModeling_indexes.csv', max_index=8, num_functions_checked=9):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{'all':[],'without_negative':[]}, key_to_rows)

        for row in tqdm(values_rows):
            key_to_rows[row[0]]['all'].append(row)
            if row[max_index] != '-1':
                key_to_rows[row[0]]['without_negative'].append(row)



        get_percentage = lambda value, row: value + (float(row[max_index]) / float(row[num_functions_checked]))

        percentages = {key: {
                                    'all': reduce(get_percentage, key_to_rows[key]['all'], 0)/len(key_to_rows[key]['all']),
                                    'without_negative': reduce(get_percentage, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative'])
                                    }
            for key in key_to_rows}


        self.x.extend(percentages.keys())
        self.y.extend([p['all'] for p in percentages.values()])


if __name__ == '__main__':
    e = Experiment_1('apache_commons-lang')()
