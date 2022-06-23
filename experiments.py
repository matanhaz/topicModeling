import json
from os.path import join, exists, isdir
from os import mkdir, listdir, remove
import math
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce

import pandas
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats

NUM_TOPICS = 30


class Experiment(ABC):
    def __init__(self, project_name, experiment_name):
        self.project_name = project_name
        self.project_path = join("projects", project_name)
        self.analysis_path = join(self.project_path, "analysis")
        self.experiment_path = join('projects', project_name, 'Experiments', experiment_name)
        self.data_path = join(self.experiment_path, 'data')
        self.results_path = join(self.experiment_path, 'results')
        self.tested_metrics_diagnosis = ["precision", "recall", "wasted" , "f-score", "expense", "t-score", "cost", "exam-score"]

        self.percentage_metrics = ["expense","t-score","exam-score"]

        if exists(join(self.data_path, f"data_all_methods_combined.csv")):
            remove(join(self.data_path, f"data_all_methods_combined.csv"))
        if exists(join(self.data_path, f"Sanity_combined.csv")):
            remove(join(self.data_path, f"Sanity_combined.csv"))
        if exists(join(self.data_path, f"average_results.csv")):
            remove(join(self.data_path, f"average_results.csv"))

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
        plt.figure(figsize=(25, 13))

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
        plt.xticks(x, rotation=15)
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

        self.x = {'TOP-K': [], 'MAP': [], 'MRR': [], 'average similarity': []}
        self.y = {'TOP-K': [], 'MAP': [], 'MRR': [], 'average similarity': []}

        self.relevant_bugs = set()
        self.best_topics = {'MRR':{} ,'average similarity':{}, 'regular':{} }


        self.combined_rows = [['method','project',  'MAP', 'MRR', 'TOP-K', 'average similarity']]

    def __call__(self):
       # map = self.MAP()
        for file in listdir(self.data_path):
            if not isdir(join(self.data_path, file)) :
                self.run(file)


        self.save_plot(self.x['TOP-K'],self.y['TOP-K'], 'num topics / method', 'average max index of all functions in results', 'Experiment_1_results_topK', f'Experiment 1 - {self.project_name}')
        self.save_plot(self.x['MAP'],self.y['MAP'], 'num topics / method', 'MAP', 'Experiment_1_results_MAP', f'Experiment 1 - {self.project_name}')
        self.save_plot(self.x['MRR'],self.y['MRR'], 'num topics / method', 'MRR', 'Experiment_1_results_MRR', f'Experiment 1 - {self.project_name}')
        self.save_plot(self.x['average similarity'],self.y['average similarity'], 'num topics / method', 'average similarity', 'Experiment_1_results_average_similarity', f'Experiment 1 - {self.project_name}')

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
        average_sim = label_to_index['average similarity']

        get_percentage_max = lambda value, row: value + (float(row[max_index]) / float(row[num_functions_checked]))
        get_percentage_mrr = lambda value, row: value + (1 / (float(row[min_index]) + 1))
        get_percentage_map = lambda value, row: value + (self._find_AP(row[all_indexes]))
        get_average_sim = lambda value, row: value + (float(row[average_sim]))

        percentages = {key: {
                'all': reduce(get_percentage_max, key_to_rows[key]['all'], 0)/len(key_to_rows[key]['all']),
                'without_negative': reduce(get_percentage_max, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                'MAP': reduce(get_percentage_map, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                'MRR':reduce(get_percentage_mrr, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                'average similarity': reduce(get_average_sim, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative'])
                } for key in key_to_rows}

        for method in percentages:
            method_name = method
            if "topic" in file_name:
                method_name += "_Topics"
            self.combined_rows.append([method_name, self.project_name, percentages[method]['MAP'], percentages[method]['MRR'],
                                       percentages[method]['without_negative'], percentages[method]['average similarity']])

        if "topic" in file_name:
            self.find_best_topics(percentages, 'MRR')
            self.find_best_topics(percentages, 'average similarity')
            # self.best_topic = [topic for topic in percentages.keys() if percentages[topic]["without_negative"] == min([val['without_negative'] for val in percentages.values()])][0]
            # self.best_topic = self.best_topic.split("_")[0]
        self.x['TOP-K'].extend(percentages.keys())
        self.y['TOP-K'].extend([p['without_negative'] for p in percentages.values()])

        self.x['MAP'].extend(percentages.keys()), self.y['MAP'].extend([p['MAP'] for p in percentages.values()])
        self.x['MRR'].extend(percentages.keys()), self.y['MRR'].extend([p['MRR'] for p in percentages.values()])
        self.x['average similarity'].extend(percentages.keys()), self.y['average similarity'].extend([p['average similarity'] for p in percentages.values()])

    def find_best_topics(self, percentages, metric_localization):
        # max_val = 0
        # best_topics = []
        # i=0
        # keys = list(percentages.keys())
        # values = list(percentages.values())
        # while i + 4 < len(keys):
        #     new_val = sum([val[metric] for val in values[i:i+5]]) / 5
        #     if new_val > max_val:
        #         max_val = new_val
        #         best_topics = keys[i:i+5]
        #     i+=1
        # self.best_topics[metric] = best_topics
        for metric in self.tested_metrics_diagnosis:
            best_topics = []
            for topic in percentages:
                if len(best_topics) < 15:
                    best_topics.append((topic, percentages[topic][metric_localization]))
                else:
                    min_topic = min(best_topics, key=lambda x:x[1])
                    if percentages[topic][metric_localization] > min_topic[1]:
                        best_topics.remove((min_topic[0], min_topic[1]))
                        best_topics.append((topic, percentages[topic][metric_localization]))

            best_topics.sort(key=lambda x:x[1],reverse=True)
            self.best_topics[metric_localization][metric] = {topics[0]:topics[1] for topics in best_topics}

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
        self.combine_tables()
        self.x = {}
        self.x = defaultdict(lambda:[], self.x)
        self.y = {}
        self.y = defaultdict(lambda:[], self.y)

        self.topics = list(range(20,601,20))


        self.is_sanity = is_sanity
        self.best_topics = best_topics

    def __call__(self):
        if self.is_sanity:
            self.run_sanity()
        else:
            self.run()

    def combine_tables(self):
        first = True
        first_sanity = True
        all_rows = []
        all_rows_sanity = []
        all_rows_positive_precision = []
        for file in listdir(self.data_path):
            if 'Sanity' not in file:
                with open(join(self.data_path, file)) as outfile:
                    rows = list(csv.reader(outfile))
                    labels_row, values_rows = rows[0], rows[1:]
                    if first:
                        all_rows.append(labels_row)
                        first = False
                    all_rows.extend(values_rows)
            else:
                with open(join(self.data_path, file)) as outfile:
                    rows = list(csv.reader(outfile))
                    labels_row, values_rows = rows[0], rows[1:]
                    if first_sanity:
                        all_rows_sanity.append(labels_row)
                        first_sanity = False
                    all_rows_sanity.extend(values_rows)



        path_to_save_into = join(self.data_path, f"data_all_methods_combined.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(all_rows)

        path_to_save_into = join(self.data_path, f"Sanity_combined.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(all_rows_sanity)

        with open(join(self.project_path, 'barinel', 'bad matrixes indexes.txt'), 'r', newline='') as file:
            bad_bug_id_list = json.load(file)['precision 0 bug id']

        for row in all_rows:
            bug_id = row[5]
            if bug_id not in bad_bug_id_list:
                all_rows_positive_precision.append(row)
        path_to_save_into = join(self.data_path, f"data_all_methods_combined_positive_precision.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(all_rows_positive_precision)

    def find_best_topics(self, key_to_rows):
        # key_to_rows => metric to a dict of topic to average
        # for now i will use only precision
        for metric in self.tested_metrics_diagnosis:
            best_topics = []
            topic_to_average = key_to_rows[metric]
            for topic in topic_to_average:
                if len(best_topics) < 15:
                    best_topics.append((topic, topic_to_average[topic]))
                else:
                    min_topic = min(best_topics, key=lambda x:x[1])
                    if topic_to_average[topic] > min_topic[1]:
                        best_topics.remove((min_topic[0], min_topic[1]))
                        best_topics.append((topic, topic_to_average[topic]))

            for method_name in self.best_topics:
                if method_name == 'regular':
                    continue
                for topic in self.best_topics[method_name][metric]:
                    self.best_topics[method_name][metric][topic] = topic_to_average[topic]

            reverse = True if metric in ['f-score', 'precision', 'recall'] else False
            best_topics.sort(key=lambda x:x[1],reverse=reverse)
            self.best_topics['regular'][metric] = {topics[0]:topics[1] for topics in best_topics}

    def compare_best_topics(self):
        rows = [['project', 'method', 'method topics', 'metric', 'metric regular topics', 'Kendall Tau value', 'RBO']]
        for method in self.best_topics:
            if method =='regular':
                continue
            for metric in self.tested_metrics_diagnosis:
                x1 = list(self.best_topics['regular'][metric].keys())
                x2 = list(self.best_topics[method][metric].keys())
                rows.append([self.project_name, method, x2, metric, x1, stats.kendalltau(x1, x2)[0], self.rbo(x1,x2)])

        path_to_save_into = join(self.data_path, f"data_topic_list_comparison.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def rbo(self, list1, list2, p=0.9):
   # tail recursive helper function
       def helper(ret, i, d):
           l1 = set(list1[:i]) if i < len(list1) else set(list1)
           l2 = set(list2[:i]) if i < len(list2) else set(list2)
           a_d = len(l1.intersection(l2))/i
           term = math.pow(p, i) * a_d
           if d == i:
               return ret + term
           return helper(ret + term, i + 1, d)
       k = max(len(list1), len(list2))
       x_k = len(set(list1).intersection(set(list2)))
       summation = helper(0, 1, k)
       return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)


    def run_sanity(self):
        for file in listdir(self.data_path):
            if 'Sanity_combined' in file:
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
                    for metric in self.tested_metrics_diagnosis:
                        key_to_rows[metric][all_similarities_values[j]][all_similarities_values[k]] += (float(row[label_to_index[metric]]) / num_of_rows)



        for key in key_to_rows:
            for test in key_to_rows[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows[key][test])


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
            for metric in self.tested_metrics_diagnosis:
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

        key_to_rows_MRR = {}
        key_to_rows_sim = {}
        key_to_rows_all ={}
        key_to_rows_MRR = defaultdict(lambda:{f"{topic}_topics_MRR":0 for topic in self.best_topics['MRR']}, key_to_rows_MRR)
        key_to_rows_sim = defaultdict(lambda:{f"{topic}_topics_avg_sim":0 for topic in self.best_topics['average similarity']}, key_to_rows_sim)
        key_to_rows_all = defaultdict(lambda:{f"{topic}":0 for topic in self.topics}, key_to_rows_all)

        average_metrics_values = {'using MRR':{metric:0 for metric in self.tested_metrics_diagnosis},
                                  'using avg sim':{metric:0 for metric in self.tested_metrics_diagnosis}}


        for i in tqdm(range(0,len(values_rows),NUM_TOPICS)):
            #
            # maxx = max([values_rows[j][4] for j in range(i,i+11)])
            for j in range(i,i+NUM_TOPICS):
                row = values_rows[j]
                topic = row[label_to_index['technique']].split('_')[0]

                for metric in self.tested_metrics_diagnosis:
                    key_to_rows_all[metric][f"{topic}"] += (float(row[label_to_index[metric]]) / num_of_rows)

                if topic in self.best_topics['MRR'] :
                    for metric in self.tested_metrics_diagnosis:
                        key_to_rows_MRR[metric][f"{topic}_topics_MRR"] += (float(row[label_to_index[metric]]) / num_of_rows)
                        average_metrics_values['using MRR'][metric] += (float(row[label_to_index[metric]]) / (num_of_rows * len(self.best_topics['MRR'])))

                if topic in self.best_topics['average similarity'] :
                    for metric in self.tested_metrics_diagnosis:
                        key_to_rows_sim[metric][f"{topic}_topics_avg_sim"] += (float(row[label_to_index[metric]]) / num_of_rows)
                        average_metrics_values['using avg sim'][metric] += (float(row[label_to_index[metric]]) / (num_of_rows * len(self.best_topics['average similarity'])))


        self.find_best_topics(key_to_rows_all)
        path_to_save_into = join(self.results_path, f"best_topics.txt")
        with open(path_to_save_into, "w", newline="") as f:
            json.dump(self.best_topics, f, indent=4)


        for key in key_to_rows_all:
            for test in key_to_rows_all[key]:
                self.x[key].append(test)
                self.y[key].append(key_to_rows_all[key][test])

        # for key in key_to_rows_MRR:
        #     for test in key_to_rows_MRR[key]:
        #         self.x[key].append(test)
        #         self.y[key].append(key_to_rows_MRR[key][test])
        #
        #     for test in key_to_rows_sim[key]:
        #         self.x[key].append(test)
        #         self.y[key].append(key_to_rows_sim[key][test])

        rows_average = [['project', 'technique'] + [metric for metric in self.tested_metrics_diagnosis]]
        rows_average.append([self.project_name,'using MRR'] + [average_metrics_values['using MRR'][metric] for metric in self.tested_metrics_diagnosis])
        rows_average.append([self.project_name,'using avg sim'] + [average_metrics_values['using avg sim'][metric] for metric in self.tested_metrics_diagnosis])
        path_to_save_into = join(self.data_path, f"average_results.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_average)

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
            if 'Sanity_combined' in file:
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
    project = 'Collections'
    if len(sys.argv) == 2:
        project = str(sys.argv[1])

    exp1 = Experiment1(project)
    exp1()
    exp2_sanity = Experiment2(project,True,0, 'old')
    exp2_sanity()
    exp2 = Experiment2(project,False, exp1.best_topics, 'old')
    exp2()
    exp2.compare_best_topics()
  #  Experiment2(project,False, exp1.best_topics, 'new')()

    # Experiment3(project,False)()
