
from os.path import join, exists
from os import mkdir, listdir

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self, project_name, experiment_name):
        self.experiment_path = join('projects', project_name, 'Experiments', experiment_name)
        self.data_path = join(self.experiment_path, 'data')
        self.results_path = join('projects', project_name, 'Experiments', 'results')


    def save_plot(self,x,y, x_label, y_label, file_name,title):
        if not exists(self.results_path):
            mkdir(self.results_path)
        plt.figure(figsize=(20, 9))
        plt.bar(x,y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.savefig(join(self.results_path, file_name), dpi=100)
        #plt.show()

    @abstractmethod
    def run(self):
        pass

class Experiment1(Experiment):
    def __init__(self, project_name):
        super().__init__(project_name, 'Experiment_1')
        self.x = []
        self.y = []

    def __call__(self):
        for file in listdir(self.data_path):
            if 'topicModeling' in file:
                self.run(file)
            else:
                self.run(file, 3, 4)

        self.save_plot(self.x,self.y, 'num topics / method', 'average max index of all functions in results', 'Experiment_1_results', 'Experiment 1')

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
                } for key in key_to_rows}

        self.x.extend(percentages.keys())
        self.y.extend([p['all'] for p in percentages.values()])


class Experiment2(Experiment):
    def __init__(self, project_name):
        super().__init__(project_name, 'Experiment_2')
        self.x = {}
        self.x = defaultdict(lambda:[], self.x)
        self.y = {}
        self.y = defaultdict(lambda:[], self.y)

    def __call__(self):
        self.run()

    def run(self):
        for file in listdir(self.data_path):
            if 'Sanity' in file:
                self._run_sanity(file)
            elif 'Topic' in file:
                self._run_topic(file)
            else:
                self._run_other(file)
        for index in self.x:
            self.save_plot(self.x[index], self.y[index], 'method', index, f'Experiment2 results - {index}', f'Experiment 2 - {index}')

    def _run_sanity(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        num_of_rows = len(values_rows) / 20

        all_combinations = ['sanity-0.6,0.4' ,'sanity-0.6,0.3','sanity-0.6,0.2','sanity-0.6,0.1',
                                          'sanity-0.7,0.4','sanity-0.7,0.3','sanity-0.7,0.2','sanity-0.7,0.1',
                                          'sanity-0.8,0.4','sanity-0.8,0.3','sanity-0.8,0.2','sanity-0.8,0.1',
                                          'sanity-0.9,0.4','sanity-0.9,0.3','sanity-0.9,0.2','sanity-0.9,0.1',
                                          'sanity-1,0.4','sanity-1,0.3','sanity-1,0.2','sanity-1,0.1']

       # all_combinations_updated = [x + " comp" for x in all_combinations] + [x + " tests" for x in all_combinations] + [x + " both" for x in all_combinations]
        all_combinations_updated = [x + " comp" for x in all_combinations]
        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{x:0 for x in all_combinations_updated},
                                  key_to_rows)

        for i in tqdm(range(0,len(values_rows),20)):

            #maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(0,20):
                row = values_rows[i+j]


                key_to_rows['precision'][all_combinations[j]+' comp'] += (float(row[5])/ num_of_rows)
               # key_to_rows['precision'][all_combinations[j] + ' tests'] += (float(row[8])/ num_of_rows)
               # key_to_rows['precision'][all_combinations[j] + ' both'] += (float(row[11])/ num_of_rows)

                key_to_rows['recall'][all_combinations[j]+' comp'] += (float(row[6])/ num_of_rows)
               # key_to_rows['recall'][all_combinations[j] + ' tests'] += (float(row[9])/ num_of_rows)
               # key_to_rows['recall'][all_combinations[j] + ' both'] += (float(row[12])/ num_of_rows)

                key_to_rows['wasted'][all_combinations[j]+' comp'] += (float(row[7])/ num_of_rows)
               # key_to_rows['wasted'][all_combinations[j] + ' tests'] += (float(row[10])/ num_of_rows)
              #  key_to_rows['wasted'][all_combinations[j] + ' both'] += (float(row[13])/ num_of_rows)



        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])

    def _run_other(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        num_of_rows = len(values_rows)

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{'precision':0, 'recall':0, 'wasted':0}, key_to_rows)

        for row in tqdm(values_rows):
            technique_name = row[0].split('_')[0]
            key_to_rows[technique_name]['precision'] += (float(row[1]) / num_of_rows)

            key_to_rows[technique_name]['recall'] += (float(row[2])/ num_of_rows)

            key_to_rows[technique_name]['wasted'] += (float(row[3])/ num_of_rows)


        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[test].append(key)
                self.y[test].append(key_to_rows[key][test])

    def _run_topic(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        num_of_rows = len(values_rows) / 11

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{'original':0 ,'comp':0, 'tests':0, 'both':0}, key_to_rows)

        for i in tqdm(range(0,len(values_rows),11)):

            maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(i,i+11):
                if values_rows[j][4] == maxx:
                    row = values_rows[j]
                    key_to_rows['precision']['original'] += (float(row[1]) / num_of_rows)
                    key_to_rows['precision']['comp'] += (float(row[4])/ num_of_rows)
                    key_to_rows['precision']['tests'] += (float(row[7])/ num_of_rows)
                    key_to_rows['precision']['both'] += (float(row[10])/ num_of_rows)

                    key_to_rows['recall']['original'] += (float(row[2])/ num_of_rows)
                    key_to_rows['recall']['comp'] += (float(row[5])/ num_of_rows)
                    key_to_rows['recall']['tests'] += (float(row[8])/ num_of_rows)
                    key_to_rows['recall']['both'] += (float(row[11])/ num_of_rows)

                    key_to_rows['wasted']['original'] += (float(row[3])/ num_of_rows)
                    key_to_rows['wasted']['comp'] += (float(row[6])/ num_of_rows)
                    key_to_rows['wasted']['tests'] += (float(row[9])/ num_of_rows)
                    key_to_rows['wasted']['both'] += (float(row[12])/ num_of_rows)
                    break


        # for row in tqdm(values_rows):
        #     key_to_rows['precision']['original'] += (float(row[1]) / num_of_rows)
        #     key_to_rows['precision']['comp'] += (float(row[4])/ num_of_rows)
        #     key_to_rows['precision']['tests'] += (float(row[7])/ num_of_rows)
        #     key_to_rows['precision']['both'] += (float(row[10])/ num_of_rows)
        #
        #     key_to_rows['recall']['original'] += (float(row[2])/ num_of_rows)
        #     key_to_rows['recall']['comp'] += (float(row[5])/ num_of_rows)
        #     key_to_rows['recall']['tests'] += (float(row[8])/ num_of_rows)
        #     key_to_rows['recall']['both'] += (float(row[11])/ num_of_rows)
        #
        #     key_to_rows['wasted']['original'] += (float(row[3])/ num_of_rows)
        #     key_to_rows['wasted']['comp'] += (float(row[6])/ num_of_rows)
        #     key_to_rows['wasted']['tests'] += (float(row[9])/ num_of_rows)
        #     key_to_rows['wasted']['both'] += (float(row[12])/ num_of_rows)

        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])

    # def _run_other(self, file_name):
    #     with open(join(self.data_path, file_name)) as outfile:
    #         rows = list(csv.reader(outfile))
    #
    #     labels_row, values_rows = rows[0], rows[1:]
    #
    #     num_of_rows = len(values_rows)
    #
    #     key_to_rows = {}
    #     key_to_rows = defaultdict(lambda:{'precision':0, 'recall':0, 'wasted':0}, key_to_rows)
    #
    #     for row in tqdm(values_rows):
    #         key_to_rows[row[0]]['precision'] += (float(row[1]) / num_of_rows)
    #
    #         key_to_rows[row[0]]['recall'] += (float(row[2])/ num_of_rows)
    #
    #         key_to_rows[row[0]]['wasted'] += (float(row[3])/ num_of_rows)
    #
    #
    #     for key in key_to_rows:
    #         for test in key_to_rows[key]:
    #             self.x[test].append(key)
    #             self.y[test].append(key_to_rows[key][test])



class Experiment3(Experiment):
    def __init__(self, project_name):
        super().__init__(project_name, 'Experiment_3')
        self.x = {}
        self.x = defaultdict(lambda:[], self.x)
        self.y = {}
        self.y = defaultdict(lambda:[], self.y)

    def __call__(self):
        self.run()

    def run(self):
        for file in listdir(self.data_path):
            if 'sanity' in file:
                pass
            elif 'Topic' in file:
                self._run_topic(file)
            else:
                pass
        for index in self.x:
            self.save_plot(self.x[index], self.y[index], 'bug index', index + " delta", f'Experiment3 results - {index}', f'Experiment 3 - {index}')

    def _run_sanity(self, file):
        pass

    def _run_topic(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        num_of_rows = len(values_rows)

        key_to_rows = {'precision': {} ,'recall': {}, 'wasted':{}}
        #key_to_rows = defaultdict(lambda:{'original':0 ,'comp':0, 'tests':0, 'both':0}, key_to_rows)

        for i in tqdm(range(0,len(values_rows),11)):
            maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(i,i+11):
                if values_rows[j][4] == maxx:
                    row = values_rows[j]
                    key_to_rows['precision'][row[-1]] = float(row[4])-float(row[1])

                    key_to_rows['recall'][row[-1]] = float(row[5])- float(row[2])

                    key_to_rows['wasted'][row[-1]] = float(row[6]) - float(row[3])

                    break


        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])

    def _run_other(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        num_of_rows = len(values_rows)

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{'precision':0, 'recall':0, 'wasted':0}, key_to_rows)

        for row in tqdm(values_rows):
            key_to_rows[row[0]]['precision'] += (float(row[1]) / num_of_rows)

            key_to_rows[row[0]]['recall'] += (float(row[2])/ num_of_rows)

            key_to_rows[row[0]]['wasted'] += (float(row[3])/ num_of_rows)


        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[test].append(key)
                self.y[test].append(key_to_rows[key][test])

    def save_plot(self,x,y, x_label, y_label, file_name,title):
        if not exists(self.results_path):
            mkdir(self.results_path)
        plt.figure(figsize=(15, 6))
        plt.plot(x,y, 'o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(join(self.results_path, file_name), dpi=100)
        #plt.show()


if __name__ == '__main__':

    Experiment2('Compress')()
