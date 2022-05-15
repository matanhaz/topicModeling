import json
from os.path import join, exists, isdir
from os import mkdir, listdir

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce

import pandas
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np

NUM_TOPICS = 30


class Experiment(ABC):
    def __init__(self, project_name, experiment_name):
        self.project_name = project_name
        self.project_path = join("projects", project_name)
        self.analysis_path = join(self.project_path, "analysis")
        self.experiment_path = join('projects', project_name, 'Experiments', experiment_name)
        self.data_path = join(self.experiment_path, 'data')
        self.results_path = join(self.experiment_path, 'results')
        self.percentage_metrics = ["expense","t-score","exam-score"]


    def label_to_index(self, labels_row):
        indexes = {}
        for index, label in enumerate(labels_row):
            indexes[label] = index
        return indexes

    def save_plot(self,x,y, x_label, y_label, file_name,title, sanity = False):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if not exists(self.results_path):
            mkdir(self.results_path)
            mkdir(join(self.results_path,"sanity"))
            #mkdir(join(self.results_path,"normal"))
        plt.figure(figsize=(22, 10))

        if sanity:
            similarities = x.copy()
            similarities.reverse()
            pos = -0.035
            for index,sim in enumerate(similarities):
                plt.bar(x=np.array(x) + pos, height=[arr[sim]for arr in y], width=0.007, label=str(sim))
                pos += 0.007
            plt.legend(title="Others Similarities")
        else:
            plt.bar(x,y, color=colors)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(x, rotation=90)
        #plt.grid(True)
        if y_label in self.percentage_metrics:
            plt.ylim(0,100)
        elif y_label != 'wasted':
            plt.ylim(0,1)

        if sanity:
            plt.savefig(join(self.results_path,"sanity", file_name), dpi=100)
        else:
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
        self.x_map = []
        self.y_map = []
        self.x_mrr = []
        self.y_mrr = []
        self.relevant_bugs = set()
        self.best_topics = []

        self.combined_rows = [['method','project',  'MAP', 'MRR', 'TOP-K']]

    def __call__(self):
       # map = self.MAP()
        for file in listdir(self.data_path):
            if not isdir(join(self.data_path, file)) :
                self.run(file)


        self.save_plot(self.x,self.y, 'num topics / method', 'average max index of all functions in results', 'Experiment_1_results_topK', f'Experiment 1 - {self.project_name}')
        self.save_plot(self.x_map,self.y_map, 'num topics / method', 'MAP', 'Experiment_1_results_MAP', f'Experiment 1 - {self.project_name}')
        self.save_plot(self.x_mrr,self.y_mrr, 'num topics / method', 'MRR', 'Experiment_1_results_MRR', f'Experiment 1 - {self.project_name}')

        with open(join(self.project_path, "Experiments", "Experiment_1", "data", "data_all_methods_combined.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(self.combined_rows)


    def run(self, file_name='topicModeling_indexes.csv'):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        label_to_index = self.label_to_index(labels_row)

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{'all':[],'without_negative':[]}, key_to_rows)

        for row in tqdm(values_rows):
            if row[label_to_index['num of files checked exist files no tests']] != '0':
                key_to_rows[row[0]]['all'].append(row)
                if row[label_to_index['num of files that changed no tests']] != '0' :
                    key_to_rows[row[0]]['without_negative'].append(row)


        max_index = label_to_index['max index exist files no tests']
        min_index = label_to_index['first index exist files no tests']
        all_indexes = label_to_index['all indexes no tests']
        num_functions_checked = label_to_index['num of files checked exist files no tests']

        get_percentage_max = lambda value, row: value + (float(row[max_index]) / float(row[num_functions_checked]))
        get_percentage_mrr = lambda value, row: value + (1 / (float(row[min_index]) + 1))
        get_percentage_map = lambda value, row: value + (self._find_AP(row[all_indexes]))

        percentages = {key: {
                'all': reduce(get_percentage_max, key_to_rows[key]['all'], 0)/len(key_to_rows[key]['all']),
                'without_negative': reduce(get_percentage_max, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                'MAP': reduce(get_percentage_map, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                'MRR':reduce(get_percentage_mrr, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative'])
                } for key in key_to_rows}

        for method in percentages:
            method_name = method
            if "topic" in file_name:
                method_name += "_Topics"
            self.combined_rows.append([method_name, self.project_name, percentages[method]['MAP'], percentages[method]['MRR'], percentages[method]['without_negative']])

        if "topic" in file_name:
            max_val = 0
            best_topics = []
            i=0
            keys = list(percentages.keys())
            values = list(percentages.values())
            while i + 4 < len(keys):
                new_val = sum([val['MRR'] for val in values[i:i+5]]) / 5
                if new_val > max_val:
                    max_val = new_val
                    best_topics = keys[i:i+5]
                i+=1
            self.best_topics = best_topics
            # self.best_topic = [topic for topic in percentages.keys() if percentages[topic]["without_negative"] == min([val['without_negative'] for val in percentages.values()])][0]
            # self.best_topic = self.best_topic.split("_")[0]
        self.x.extend(percentages.keys())
        self.y.extend([p['without_negative'] for p in percentages.values()])

        self.x_map.extend(percentages.keys()), self.y_map.extend([p['MAP'] for p in percentages.values()])
        self.x_mrr.extend(percentages.keys()), self.y_mrr.extend([p['MRR'] for p in percentages.values()])

    def get_relevant_bugs(self):
        first = True
        for file in listdir(join(self.data_path, 'methods')):
            df = pandas.read_parquet(path=join(self.data_path, "methods" ,file))
            bug_to_func_and_similarity = df.to_dict()["bugs"]
            if first:
                first = False
                self.relevant_bugs = set(bug_to_func_and_similarity.keys())
            else:
                self.relevant_bugs = self.relevant_bugs & set(bug_to_func_and_similarity.keys())

    def MAP(self):

        self.get_relevant_bugs()
        for file in listdir(join(self.data_path, 'methods')):
            map = 0
            mrr = 0
            with open(join(self.project_path, "analysis", "bug_to_commit_that_solved.txt")) as outfile:
                bugs = json.load(outfile)["bugs to commit"]
            df = pandas.read_parquet(path=join(self.data_path, "methods" ,file))
            bug_to_func_and_similarity = df.to_dict()["bugs"]

            df = pandas.read_parquet(
                path=join(self.analysis_path, "commitId to all functions")
            )
            commit_to_exist_functions = df.to_dict()["commit id"]

            count_works = 0
            for bug in tqdm(bugs):
                #len(bug["function that changed"]) > 10 or
                if bug["bug id"] not in self.relevant_bugs:
                    continue

                indexes = self._find_indexes(bug, bug_to_func_and_similarity, commit_to_exist_functions[bug["commit number"]]["all functions"].tolist())
                if indexes != []:
                    count_works += 1
                    a_p = self._find_AP(indexes)
                    map += a_p
                    min_index = min(indexes) +1
                    mrr += (1/min_index)
                # else:
                #     print(bug['bug id'])
                #     print(bug['function that changed'])
                #     print(bug['hexsha'])
            map /= count_works
            mrr /= count_works
            self.x_map.append(file), self.y_map.append(map)
            self.x_mrr.append(file), self.y_mrr.append(mrr)


    def _find_indexes(self, bug, bug_to_func_and_similarity, commit_to_exist_functions):
        # filter only exists functions
        func_and_similarity_of_bug = (
            bug_to_func_and_similarity[bug["bug id"]].tolist().copy()
        )
        for i in range(len(func_and_similarity_of_bug)):
            func_and_similarity_of_bug[i] = func_and_similarity_of_bug[i].tolist(
            )
        # now im finiding the index only on the list of existing functions in the commit
        exist_funcs_with_similarity = []

        for func_exist in commit_to_exist_functions:
            for func_and_similarity in func_and_similarity_of_bug:
                if func_exist == func_and_similarity[0]:
                    exist_funcs_with_similarity.append(func_and_similarity)
                    func_and_similarity_of_bug.remove(func_and_similarity)
                    break
        exist_funcs_with_similarity.sort(key=lambda x: x[1], reverse=True)

        #find all indexes of relevant functions
        indexes = set()
        for func in bug["function that changed"]:
            index = 0
            for exist_func_and_similarity in exist_funcs_with_similarity:
                if func["function name"] == exist_func_and_similarity[0]:
                    indexes.add(index)
                    break
                index += 1
        indexes = list(indexes)
        indexes.sort()
        return indexes

    def _find_AP(self,indexes):
        filtered = indexes[1:-1].split(', ')
        a_p = 0
        for i, index in enumerate(filtered):
            p_k = (i+1) / (int(index)+1)
            a_p += (p_k / len(indexes))



        return a_p




class Experiment2(Experiment):
    def __init__(self, project_name, is_sanity, best_topics, type_of_exp):
        exp = 'Experiment_2' if type_of_exp == 'old' else 'Experiment_4'
        super().__init__(project_name, exp)
        self.x = {}
        self.x = defaultdict(lambda:[], self.x)
        self.y = {}
        self.y = defaultdict(lambda:[], self.y)
        self.is_sanity = is_sanity
        self.tested_metrics = ["precision", "recall", "wasted" , "f-score","expense","t-score", "cost", "exam-score"]
        self.best_topics = best_topics

    def __call__(self):
        if self.is_sanity:
            self.run_sanity()
        else:
            self.run()



    def run_sanity(self):
        for file in listdir(self.data_path):
            if 'Sanity' in file:
                self._run_sanity(file)
            else:
                pass

        for index in self.x:
            self.save_plot(self.x[index], self.y[index], 'Actual Bug Similarities', index, f'Experiment2 results - {index} - sanity', f'Experiment 2 Sanity - {index} - {self.project_name}', sanity = True)


    def run(self):
        for file in listdir(self.data_path):
            if 'Sanity' in file or 'combined' in file:
                pass
            elif 'Topic' in file or 'original' in file:
                self._run_with_topics(file)
            else:
                self._run_all_methods(file)
        for index in self.x:
            self.save_plot(self.x[index], self.y[index], 'method', index, f'Experiment2 results - {index}', f'Experiment 2 - {index} - {self.project_name}')

    def _run_sanity(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        label_to_index = self.label_to_index(labels_row)
        num_of_rows = len(values_rows) / 121


        all_similarities_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

       # all_combinations_updated = [x + " comp" for x in all_combinations] + [x + " tests" for x in all_combinations] + [x + " both" for x in all_combinations]
        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{x:{y:0 for y in all_similarities_values} for x in all_similarities_values},
                                  key_to_rows)


        for i in tqdm(range(0,len(values_rows),121)):

            #maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(len(all_similarities_values)):
                for k in range(len(all_similarities_values)):
                    row = values_rows[i+j*11+k]
                    for metric in self.tested_metrics:
                        key_to_rows[metric][all_similarities_values[j]][all_similarities_values[k]] += (float(row[label_to_index[metric]]) / num_of_rows)

                    # key_to_rows['precision'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[5])/ num_of_rows)
                    #
                    #
                    # key_to_rows['recall'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[6])/ num_of_rows)
                    #
                    #
                    # key_to_rows['wasted'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[7])/ num_of_rows)
                    #
                    # f_score = (float(row[5]) * float(row[6]) * 2) / (float(row[5]) + float(row[6])) if (float(row[5]) + float(row[6])) != 0 else 0
                    #
                    # key_to_rows['f-score'][all_similarities_values[j]][all_similarities_values[k]] += (f_score/ num_of_rows)
                    # key_to_rows['precision'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[label_to_index['precision-sigmoid']])/ num_of_rows)
                    #
                    # key_to_rows['recall'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[label_to_index['recall-sigmoid']])/ num_of_rows)
                    #
                    # key_to_rows['wasted'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[label_to_index['wasted-sigmoid']])/ num_of_rows)
                    #
                    # key_to_rows['f-score'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[label_to_index['f-score-sigmoid']])/ num_of_rows)




        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])

    # def _run_other(self, file_name):
    #     with open(join(self.data_path, file_name)) as outfile:
    #         rows = list(csv.reader(outfile))
    #
    #     labels_row, values_rows = rows[0], rows[1:]
    #     label_to_index = self._label_to_index(labels_row)
    #     num_of_rows = len(values_rows)
    #
    #     key_to_rows = {}
    #     key_to_rows = defaultdict(lambda:{'precision':0, 'recall':0, 'wasted':0, 'f-score':0}, key_to_rows)
    #
    #     for row in tqdm(values_rows):
    #         technique_name = row[0].split('_')[0]
    #         key_to_rows[technique_name]['precision'] += (float(row[label_to_index['precision']]) / num_of_rows)
    #
    #         key_to_rows[technique_name]['recall'] += (float(row[label_to_index['recall']])/ num_of_rows)
    #
    #         key_to_rows[technique_name]['wasted'] += (float(row[label_to_index['wasted']])/ num_of_rows)
    #
    #         key_to_rows[technique_name]['f-score'] += (float(row[label_to_index['f-score']])/ num_of_rows)
    #
    #
    #     for key in key_to_rows:
    #         for test in key_to_rows[key]:
    #             self.x[test].append(key)
    #             self.y[test].append(key_to_rows[key][test])
    #
    # def _run_topic(self, file_name):
    #     with open(join(self.data_path, file_name)) as outfile:
    #         rows = list(csv.reader(outfile))
    #
    #     labels_row, values_rows = rows[0], rows[1:]
    #     label_to_index = self._label_to_index(labels_row)
    #     num_of_rows = len(values_rows) / 11
    #
    #     key_to_rows = {}
    #     #key_to_rows = defaultdict(lambda:{'original':0 ,'comp':0, 'tests':0, 'both':0}, key_to_rows)
    #     key_to_rows = defaultdict(lambda:{'original':0 ,'sigmuid':0 ,'multiply':0}, key_to_rows)
    #     for i in tqdm(range(0,len(values_rows),11)):
    #
    #         maxx = max([values_rows[j][4] for j in range(i,i+11)])
    #         for j in range(i,i+11):
    #             if values_rows[j][4] == maxx:
    #                 row = values_rows[j]
    #                 key_to_rows['precision']['original'] += (float(row[label_to_index['original precision']]) / num_of_rows)
    #                 key_to_rows['precision']['sigmuid'] += (float(row[label_to_index['precision-sigmoid']])/ num_of_rows)
    #                 #key_to_rows['precision']['tests'] += (float(row[7])/ num_of_rows)
    #                 key_to_rows['precision']['multiply'] += (float(row[label_to_index['precision-multiply']])/ num_of_rows)
    #
    #                 key_to_rows['recall']['original'] += (float(row[label_to_index['original recall']])/ num_of_rows)
    #                 key_to_rows['recall']['sigmuid'] += (float(row[label_to_index['recall-sigmoid']])/ num_of_rows)
    #                 #key_to_rows['recall']['tests'] += (float(row[8])/ num_of_rows)
    #                 key_to_rows['recall']['multiply'] += (float(row[label_to_index['recall-multiply']])/ num_of_rows)
    #
    #                 key_to_rows['wasted']['original'] += (float(row[label_to_index['original wasted']])/ num_of_rows)
    #                 key_to_rows['wasted']['sigmuid'] += (float(row[label_to_index['wasted-sigmoid']])/ num_of_rows)
    #                 #key_to_rows['wasted']['tests'] += (float(row[9])/ num_of_rows)
    #                 key_to_rows['wasted']['multiply'] += (float(row[label_to_index['wasted-multiply']])/ num_of_rows)
    #
    #
    #                 # f_score_original = (float(row[1]) * float(row[2]) * 2) / (float(row[1]) + float(row[2])) if (float(row[1]) + float(row[2])) != 0 else 0
    #                 # f_score_sigmuid = (float(row[4]) * float(row[5]) * 2) / (float(row[4]) + float(row[5])) if (float(row[4]) + float(row[5])) != 0 else 0
    #                 # f_score_multiply = (float(row[10]) * float(row[11]) * 2) / (float(row[10]) + float(row[11])) if (float(row[10]) + float(row[11])) != 0 else 0
    #
    #                 key_to_rows['f-score']['original'] += (float(row[label_to_index['f-score-original']])/ num_of_rows)
    #                 key_to_rows['f-score']['sigmuid'] += (float(row[label_to_index['f-score-sigmoid']])/ num_of_rows)
    #                 key_to_rows['f-score']['multiply'] += (float(row[label_to_index['f-score-multiply']])/ num_of_rows)
    #
    #                 break
    #
    #     for key in key_to_rows:
    #         for test in key_to_rows[key]:
    #             self.x[key].append(test)
    #             self.y[key].append(key_to_rows[key][test])

    def _run_all_methods(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))
        file_name = file_name.split('.')[0]

        labels_row, values_rows = rows[0], rows[1:]
        label_to_index = self.label_to_index(labels_row)

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{file_name:0}, key_to_rows)
        num_of_rows = len(values_rows)

        for row in tqdm(values_rows):
            for metric in self.tested_metrics:
                key_to_rows[metric][file_name] += (float(row[label_to_index[metric]]) / num_of_rows)


        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])


    def _run_with_topics(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))
        file_name = file_name.split('.')[0]
        labels_row, values_rows = rows[0], rows[1:]
        label_to_index = self.label_to_index(labels_row)
        num_of_rows = len(values_rows) / NUM_TOPICS

        key_to_rows = {}
        #key_to_rows = defaultdict(lambda:{'original':0 ,'comp':0, 'tests':0, 'both':0}, key_to_rows)
        key_to_rows = defaultdict(lambda:{file_name + f"_{topic}":0 for topic in self.best_topics}, key_to_rows)
        for i in tqdm(range(0,len(values_rows),NUM_TOPICS)):
            #
            # maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(i,i+NUM_TOPICS):
                row = values_rows[j]
                topic = row[label_to_index['technique']].split('_')[0]
                if topic in self.best_topics :
                    for metric in self.tested_metrics:
                        key_to_rows[metric][file_name + f"_{topic}"] += (float(row[label_to_index[metric]]) / num_of_rows)

                 #   break

        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])



class Experiment3(Experiment):
    def __init__(self, project_name, is_sanity):
        super().__init__(project_name, 'Experiment_3')
        self.all_percentages_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.x = {}
        self.x = defaultdict(lambda:self.all_percentages_values, self.x)
        self.y = {}
        self.y = defaultdict(lambda: {x:{} for x in self.all_percentages_values}, self.y)
        self.is_sanity = is_sanity
        self.techniques = []

    def save_plot(self,x,y, x_label, y_label, file_name,title, sanity = False):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if not exists(self.results_path):
            mkdir(self.results_path)
        plt.figure(figsize=(15, 6))

        if sanity:
            # similarities = x.copy()
            # similarities.reverse()
            # pos = -0.035
            # for index,sim in enumerate(similarities):
            #     plt.bar(x=np.array(x) + pos, height=[arr[sim]for arr in y], width=0.007, label=str(sim))
            #     pos += 0.007
            # plt.legend(title="Others Similarities")
            pass
        else:
            percentages = x.copy()

            pos = -0.015
            width = 0.06/len(self.techniques)
            for technique in self.techniques:
                plt.bar(x=np.array(x) + pos, height=[y[key][technique] for key in y], width=width, label=str(technique))
                pos += width
            plt.legend(title="Technique")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(x)
        #plt.grid(True)
        if y_label != 'wasted':
            plt.ylim(0,1)
        plt.savefig(join(self.results_path, file_name), dpi=100)
        #plt.show()

    def __call__(self):
        if self.is_sanity:
            pass
            self.run_sanity()
        else:
            self.run()

    def run_sanity(self):
        for file in listdir(self.data_path):
            if 'Sanity' in file:
                self._run_sanity(file)
            else:
                pass

        for index in self.x:
            self.save_plot(self.x[index], self.y[index], 'Actual Bug Similarities', index, f'Experiment2 results - {index} - sanity', f'Experiment 2 Sanity - {index} - {self.project_name}', sanity = True)


    def run(self):
        for file in listdir(self.data_path):
            if 'Sanity' in file :
                pass
            elif 'Topic' in file:
                self._run_topic(file)
            else:
                self._run_other(file)
        for index in self.x:
            self.save_plot(self.x[index], self.y[index], 'Original Score Percentage', index, f'Experiment3 results - {index}', f'Experiment 3 - {index} - {self.project_name}')


    def _run_sanity(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        num_of_rows = len(values_rows) / 121


        all_similarities_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

       # all_combinations_updated = [x + " comp" for x in all_combinations] + [x + " tests" for x in all_combinations] + [x + " both" for x in all_combinations]
        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{x:{y:0 for y in all_similarities_values} for x in all_similarities_values},
                                  key_to_rows)

        for i in tqdm(range(0,len(values_rows),121)):

            #maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(len(all_similarities_values)):
                for k in range(len(all_similarities_values)):
                    row = values_rows[i+j*11+k]


                    key_to_rows['precision'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[5])/ num_of_rows)
                   # key_to_rows['precision'][all_combinations[j] + ' tests'] += (float(row[8])/ num_of_rows)
                   # key_to_rows['precision'][all_combinations[j] + ' both'] += (float(row[11])/ num_of_rows)

                    key_to_rows['recall'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[6])/ num_of_rows)
                   # key_to_rows['recall'][all_combinations[j] + ' tests'] += (float(row[9])/ num_of_rows)
                   # key_to_rows['recall'][all_combinations[j] + ' both'] += (float(row[12])/ num_of_rows)

                    key_to_rows['wasted'][all_similarities_values[j]][all_similarities_values[k]] += (float(row[7])/ num_of_rows)
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
        technique_name = values_rows[0][0].split('_')[0]
        self.techniques.append(technique_name)
        num_of_rows = len(values_rows)/11

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{float(x):{y:0 for y in [technique_name]} for x in self.all_percentages_values},
                                  key_to_rows)

        for row in tqdm(values_rows):

            key_to_rows['precision'][float(row[5])][technique_name] += (float(row[1]) / num_of_rows)

            key_to_rows['recall'][float(row[5])][technique_name] += (float(row[2])/ num_of_rows)

            key_to_rows['wasted'][float(row[5])][technique_name] += (float(row[3])/ num_of_rows)


        for key in key_to_rows:
            for test in key_to_rows[key]:
                tmp = self.x[key]
                self.y[key][test][technique_name] = key_to_rows[key][test][technique_name]

    def _run_topic(self, file_name):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        technique_name = "Topic Modeling"
        self.techniques.append(technique_name)

        num_of_rows = len(values_rows) / (len(self.all_percentages_values) * 11)

        key_to_rows = {}

        key_to_rows = defaultdict(lambda:{float(x):{y:0 for y in [technique_name]} for x in self.all_percentages_values},
                                  key_to_rows)


        for i in tqdm(range(0,len(values_rows),11)):
            maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(i,i+11):
                if values_rows[j][4] == maxx:
                    row = values_rows[j]
                    key_to_rows['precision'][float(row[14])]['Topic Modeling'] += (float(row[4])/ num_of_rows)

                    key_to_rows['recall'][float(row[14])]['Topic Modeling'] += (float(row[5])/ num_of_rows)

                    key_to_rows['wasted'][float(row[14])]['Topic Modeling'] += (float(row[6])/ num_of_rows)
                    break



        for key in key_to_rows:
            for test in key_to_rows[key]:
                tmp = self.x[key]
                self.y[key][test][technique_name] = key_to_rows[key][test][technique_name]



import sys
if __name__ == '__main__':
    project = 'Codec'
    if len(sys.argv) == 2:
        project = str(sys.argv[1])
    exp1 = Experiment1(project)
    exp1()
    Experiment2(project,True,0)()
    Experiment2(project,False, exp1.best_topics, 'old')()
    Experiment2(project,False, exp1.best_topics, 'new')()

    # Experiment3(project,False)()
