import csv
import json
from collections import defaultdict
from functools import reduce
from os.path import join, exists, isdir, isfile, islink
from os import mkdir, listdir, rename, unlink, remove, getcwd
import shutil
from pathlib import Path

import sys
import pandas as pd

from sfl.Diagnoser.diagnoserUtils import (
    read_json_planning_file,
    read_json_planning_instance,
    write_json_planning_file,
)
from sfl.Diagnoser.Diagnosis_Results import Diagnosis_Results
from sfl.Diagnoser.Experiment_Data import Experiment_Data


rows_combined_methods = []
rows_combined_methods.append(["project", "technique","precision", "recall", "wasted","bug id","original score percentage","f-score","expense",
                              "t-score", "cost", "exam-score", ])


class BarinelTester:
    def __init__(self, project_name, test_type, local):
        self.project_name = project_name
        self.epsilon = 0.01
        self.rows = []
        self.rows.append(["project", "technique","precision", "recall", "wasted","bug id","original score percentage","f-score","expense",
                              "t-score", "cost", "exam-score", ])
        self.rows_all_divisions = []
        self.rows_all_divisions.append(["project", "technique","precision", "recall", "wasted","bug id","original score percentage","f-score","expense",
                              "t-score", "cost", "exam-score", ])

        self.test_type = test_type
        self.optimal_original_score_percentage = 0.2
        self.percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


        if local:
            self.project_path = join(str(Path(__file__).parents[1]),"projects",project_name)
        else:
            self.project_path = join(str(Path(getcwd())),"projects",project_name)

        self.experiment1_path = join(self.project_path, 'Experiments', 'Experiment_1')
        self.data_path = join(self.experiment1_path, 'data')
        self.best_topics = []
        self.find_best_topics()

        self.best_topics = ['400' ,'420' ,'440' ,'460' ,'480' ,'500' ,'520' ,'540' ,'560' ,'580' ,'600']

        self.experiment2_path = join(self.project_path,'Experiments', 'Experiment_2')
        self.experiment3_path = join(self.project_path,'Experiments', 'Experiment_3')
        self.mapping = {}
        self.prepare_dir()
        self.prepare_matrixes()

        # used for testing
        self.low_precision = []
        self.high_precision = []

    def prepare_dir(self):
        if not exists(join(self.project_path, "barinel")):
            mkdir(join(self.project_path, "barinel"))

        # generate matrixes somehow, need amir's code

    def prepare_matrixes(self):
        with open(join(self.project_path, "analysis", "bug_to_commit_that_solved.txt")) as f:
            data = json.loads(f.read())["bugs to commit"]  # array of dicts, each represent a bug that i discovered

       # bugz = [] # debug bug id
       # for bb in data:
       #     bugz.append(bb["bug id"].split("-")[1])
       # print(bugz)

        old_path_matrixes = join\
            (self.project_path, "barinel", "matrixes_before_change")
        new_path_matrixes = join\
            (self.project_path, "barinel", "matrixes")

        """move avtive-bugs.csv"""
        if not exists(join(self.project_path, "barinel",'active-bugs.csv' )):

            rename(join(old_path_matrixes,'active-bugs.csv'),join(self.project_path, "barinel",'active-bugs.csv' ))


        """clear matrixes dir"""
        for file in listdir(new_path_matrixes):
            file_path = join\
                (new_path_matrixes, file)
            try:
                if isfile(file_path) or islink(file_path):
                    unlink(file_path)
                elif isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

        """copy files"""
        for file in listdir(old_path_matrixes):
            shutil.copy(
                join
                (old_path_matrixes, file),
                join
                (new_path_matrixes, file),
            )

        """remove json"""
        for file in listdir(new_path_matrixes):
            new_name = file.split(".")[0]
            rename(join
                      (new_path_matrixes, file), join
            (new_path_matrixes, new_name))

        """read csv"""
        df = pd.read_csv(join(self.project_path, "barinel",'active-bugs.csv')).to_dict()
        all_matrixes_in_dir = []
        for i in range(len(df["bug.id"])):
            if str(df["bug.id"][i]) in listdir(new_path_matrixes):
                all_matrixes_in_dir.append([str(df["bug.id"][i]), str(df["report.id"][i].split("-")[1])])  # (bug file index, bug actual id in jira)

        """add to each matrix his hexsha"""
        all_matrixes_in_dir_filtered = []
        for m in all_matrixes_in_dir:

            for bug in data:
                if bug["bug id"].split("-")[1] == m[1]:
                    m.append(bug["hexsha"])
                    all_matrixes_in_dir_filtered.append(m)
                    self.mapping[m[2]] = m[1]
                    break
            else:
                remove(join(new_path_matrixes, m[0]))



        """treansfer matrixes to a new location"""
        counter = 1
        for m in all_matrixes_in_dir_filtered:
            if isfile(join(new_path_matrixes, m[2])):
                try:
                    unlink(join(new_path_matrixes, m[0]))
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (join(new_path_matrixes, m[0]), e))

                # rename(join(new_path_matrixes, m[0]),
                #        join(new_path_matrixes, f"{m[2]}_{str(counter)}"))
                # counter += 1
            else:
                rename(join
                          (new_path_matrixes, m[0]), join
                (new_path_matrixes, m[2]))

    def write_rows(self):
        if not exists(self.experiment2_path):
            mkdir(self.experiment2_path)
            mkdir(join(self.experiment2_path, "data"))

        if not exists(self.experiment3_path):
            mkdir(self.experiment3_path)
            mkdir(join(self.experiment3_path, "data"))


        path_to_save_into = join(self.experiment2_path, "data", f"{self.test_type}.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)

        path_to_save_into = join(self.experiment3_path, "data", f"{self.test_type}.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows_all_divisions)

    def _fill_row(self, diagnosis, matrix_name, percentage, is_sanity, *args):
        if is_sanity:
            return [
                            self.project_name,
                            args[0], #good sim
                            args[1], # bad sim
                            diagnosis["precision"],
                            diagnosis["recall"],
                            diagnosis["wasted"],
                            self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                            percentage,
                            diagnosis["fscore"],
                            diagnosis["expense"],
                            diagnosis["tscore"],
                            diagnosis["cost"],
                            diagnosis["exam_score"],
                        ]
        else:
            return [
                            self.project_name,
                            args[0], # technique
                            diagnosis["precision"],
                            diagnosis["recall"],
                            diagnosis["wasted"],
                            self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                            percentage,
                            diagnosis["fscore"],
                            diagnosis["expense"],
                            diagnosis["tscore"],
                            diagnosis["cost"],
                            diagnosis["exam_score"],
                        ]

    def label_to_index(self, labels_row):
        indexes = {}
        for index, label in enumerate(labels_row):
            indexes[label] = index
        return indexes

    def find_best_topics(self, file_name='topicModeling_indexes.csv'):
        with open(join(self.data_path, file_name)) as outfile:
            rows = list(csv.reader(outfile))

        labels_row, values_rows = rows[0], rows[1:]

        label_to_index = self.label_to_index(labels_row)

        key_to_rows = {}
        key_to_rows = defaultdict(lambda:{'all':[],'without_negative':[]}, key_to_rows)

        for row in values_rows:
            if row[label_to_index['num of files checked exist files no tests']] != '0':
                key_to_rows[row[0]]['all'].append(row)
                if row[label_to_index['num of files that changed no tests']] != '0' :
                    key_to_rows[row[0]]['without_negative'].append(row)


        max_index = label_to_index['max index exist files no tests']
        min_index = label_to_index['first index exist files no tests']
        all_indexes = label_to_index['all indexes no tests']
        num_functions_checked = label_to_index['num of files checked exist files no tests']

        #get_percentage_max = lambda value, row: value + (float(row[max_index]) / float(row[num_functions_checked]))
        get_percentage_mrr = lambda value, row: value + (1 / (float(row[min_index]) + 1))
        #get_percentage_map = lambda value, row: value + (self._find_AP(row[all_indexes]))

        percentages = {key: {
                #'all': reduce(get_percentage_max, key_to_rows[key]['all'], 0)/len(key_to_rows[key]['all']),
                #'without_negative': reduce(get_percentage_max, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                #'MAP': reduce(get_percentage_map, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative']),
                'MRR':reduce(get_percentage_mrr, key_to_rows[key]['without_negative'], 0)/len(key_to_rows[key]['without_negative'])
                } for key in key_to_rows}

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
            #self.best_topic = self.best_topic.split("_")[0]




class BarinelTesterSanity(BarinelTester):
    def __init__(self, project_name, local):
        super().__init__(project_name, "Sanity",local)
        self.high_similarity = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.low_similarity = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.rows = []
        self.rows_all_divisions = []
        self.rows.append(
            ["project","real bug diagnose sim", "unreal bug diagnose sim","precision", "recall", "wasted","bug id","original score percentage","f-score","expense",
                              "t-score", "cost", "exam-score", ]
        )
        self.rows_all_divisions.append(
            ["project","real bug diagnose sim", "unreal bug diagnose sim","precision", "recall", "wasted","bug id","original score percentage","f-score","expense",
                              "t-score", "cost", "exam-score", ]
        )



    def diagnose(self, matrix_name):
        # getting basic values

        def diagnose_sanity(matrix_name, exp_type, good_sim, bad_sim, OriginalScorePercentage):
            ei = read_json_planning_file(matrix_name,exp_type,'sanity', good_sim=good_sim, bad_sim=bad_sim, OriginalScorePercentage = OriginalScorePercentage)
            ei.diagnose()
            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis

        # diagnoses = diagnose_sanity(matrix_name, "normal", 0, 0, 0)
        # original_precision = diagnoses["precision"]
        # original_recall = diagnoses["recall"]
        # original_wasted = diagnoses["wasted"]
        # original_fscore = diagnoses["fscore"]

        # if original_precision < self.epsilon or original_precision > 1 - self.epsilon:
        #     return
        for percentage in self.percentages:
            for good_sim in self.high_similarity:
                for bad_sim in self.low_similarity:
                    diagnosis_comp = diagnose_sanity(matrix_name, "CompSimilarity", good_sim, bad_sim, percentage)
                    # diagnosis_tests = diagnose_sanity(matrix_name, "TestsSimilarity", good_sim, bad_sim, percentage)
                    #diagnosis_both = diagnose_sanity(matrix_name, "BothSimilarities", good_sim, bad_sim, percentage)
   # def _fill_row(self, diagnosis_comp, matrix_name, percentage, True, good_sim, bad_sim):
                    self.rows_all_divisions.append(self._fill_row(diagnosis_comp, matrix_name, percentage, True, good_sim, bad_sim))
                    # self.rows_all_divisions.append(
                    #     [
                    #         good_sim,
                    #         bad_sim,
                    #         diagnosis_comp["precision"],
                    #         diagnosis_comp["recall"],
                    #         diagnosis_comp["wasted"],
                    #         self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                    #         percentage,
                    #         diagnosis_comp["fscore"],
                    #         diagnosis_comp["expense"],
                    #         diagnosis_comp["tscore"],
                    #         diagnosis_comp["cost"],
                    #         diagnosis_comp["exam_score"],
                    #     ]
                    # )
                    if percentage == self.optimal_original_score_percentage:
                        self.rows.append(self._fill_row(diagnosis_comp, matrix_name, percentage, True, good_sim, bad_sim))
                    #     self.rows.append(
                    #     [
                    #         good_sim,
                    #         bad_sim,
                    #         diagnosis_comp["precision"],
                    #         diagnosis_comp["recall"],
                    #         diagnosis_comp["wasted"],
                    #         self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                    #         percentage,
                    #         diagnosis_comp["fscore"],
                    #         diagnosis_comp["expense"],
                    #         diagnosis_comp["tscore"],
                    #         diagnosis_comp["cost"],
                    #         diagnosis_comp["exam_score"],
                    #     ]
                    # )


class BarinelTesterTopicModeling(BarinelTester):
    def __init__(self, project_name, topics_range, local):
        super().__init__(project_name, "TopicModeling", local)
        self.topics = topics_range

    def diagnose(self, matrix_name):
        # getting basic values

        def diagnose_real(matrix_name, exp_type, topic,OriginalScorePercentage):
            ei = read_json_planning_file(matrix_name, exp_type,'topic modeling', num_topics=topic, Project_name=self.project_name, OriginalScorePercentage=OriginalScorePercentage)
            ei.diagnose()

            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis

        tmp_rows = []
        for percentage in self.percentages:
            for topic in self.topics:
                diagnosis_comp = diagnose_real(matrix_name, "CompSimilarity", topic, percentage)
                row = self._fill_row(diagnosis_comp, matrix_name, percentage, False, f"{topic}_sigmuid")
                #print('finished topics ', topic)
                self.rows_all_divisions.append(row)
                # self.rows_all_divisions.append(
                #         [
                #             topic,
                #             diagnosis_comp["precision"],
                #             diagnosis_comp["recall"],
                #             diagnosis_comp["wasted"],
                #             self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                #             percentage,
                #             diagnosis_comp["fscore"],
                #             diagnosis_comp["expense"],
                #             diagnosis_comp["tscore"],
                #             diagnosis_comp["cost"],
                #             diagnosis_comp["exam_score"],
                #         ]
                #     )
                if percentage == self.optimal_original_score_percentage:
                    self.rows.append(row)
                    # self.rows.append(
                    #     [
                    #         topic,
                    #         diagnosis_comp["precision"],
                    #         diagnosis_comp["recall"],
                    #         diagnosis_comp["wasted"],
                    #         self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                    #         percentage,
                    #         diagnosis_comp["fscore"],
                    #         diagnosis_comp["expense"],
                    #         diagnosis_comp["tscore"],
                    #         diagnosis_comp["cost"],
                    #         diagnosis_comp["exam_score"],
                    #     ]
                    # )
                    if str(topic) in self.best_topics:
                        tmp_rows.append(row.copy())
                        rows_combined_methods.append(row.copy())
                    # rows_combined_methods.append(
                    #     [
                    #         f"{topic}_sigmuid",
                    #         diagnosis_comp["precision"],
                    #         diagnosis_comp["recall"],
                    #         diagnosis_comp["wasted"],
                    #         self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                    #         percentage,
                    #         diagnosis_comp["fscore"],
                    #         diagnosis_comp["expense"],
                    #         diagnosis_comp["tscore"],
                    #         diagnosis_comp["cost"],
                    #         diagnosis_comp["exam_score"],
                    #     ]
                    # )
        # best_row = max(tmp_rows, key=lambda x:x[2])
        # best_row[1] = best_row[1].split('_')[1]
        # rows_combined_methods.append(best_row)

class BarinelTesterMultiply(BarinelTester):
    def __init__(self, project_name, topics_range, local, ):
        super().__init__(project_name, "Multiply", local)
        self.topics = topics_range


    def diagnose(self, matrix_name):
        # getting basic values

        def diagnose_real(matrix_name, exp_type, topic,OriginalScorePercentage):
            ei = read_json_planning_file(matrix_name, exp_type,'topic modeling', num_topics=topic, Project_name=self.project_name, OriginalScorePercentage=OriginalScorePercentage)
            ei.diagnose()

            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis
        tmp_rows = []
        for percentage in self.percentages:
            for topic in self.topics:

                diagnosis_both = diagnose_real(matrix_name, "BothSimilarities", topic, percentage)
                row = self._fill_row(diagnosis_both, matrix_name, percentage, False, f"{topic}_multiply")
                #print('finished topics ', topic)
                self.rows_all_divisions.append(row)
                # self.rows_all_divisions.append(
                #         [
                #             topic,
                #             diagnosis_both["precision"],
                #             diagnosis_both["recall"],
                #             diagnosis_both["wasted"],
                #             self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                #             percentage,
                #             diagnosis_both["fscore"],
                #             diagnosis_both["expense"],
                #             diagnosis_both["tscore"],
                #             diagnosis_both["cost"],
                #             diagnosis_both["exam_score"],
                #         ]
                #     )
                if percentage == self.optimal_original_score_percentage:
                    self.rows.append(row)
                    # self.rows.append(
                    #     [
                    #          topic,
                    #         diagnosis_both["precision"],
                    #         diagnosis_both["recall"],
                    #         diagnosis_both["wasted"],
                    #         self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                    #         percentage,
                    #         diagnosis_both["fscore"],
                    #         diagnosis_both["expense"],
                    #         diagnosis_both["tscore"],
                    #         diagnosis_both["cost"],
                    #         diagnosis_both["exam_score"],
                    #     ]
                    # )
                    if str(topic) in self.best_topics:
                        tmp_rows.append(row.copy())
                        rows_combined_methods.append(row.copy())
                    # rows_combined_methods.append(
                    #     [
                    #          f"{topic}_multiply",
                    #         diagnosis_both["precision"],
                    #         diagnosis_both["recall"],
                    #         diagnosis_both["wasted"],
                    #         self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                    #         percentage,
                    #         diagnosis_both["fscore"],
                    #         diagnosis_both["expense"],
                    #         diagnosis_both["tscore"],
                    #         diagnosis_both["cost"],
                    #         diagnosis_both["exam_score"],
                    #     ]
                    # )
        # best_row = max(tmp_rows, key=lambda x:x[2])
        # best_row[1] = best_row[1].split('_')[1]
        # rows_combined_methods.append(best_row)



class BarinelTesterOriginalMethod(BarinelTester):
    def __init__(self, project_name, local):
        super().__init__(project_name, "Original", local)


    def diagnose(self, matrix_name):
        # getting basic values

        def diagnose_real(matrix_name, exp_type, topic,OriginalScorePercentage):
            ei = read_json_planning_file(matrix_name, exp_type,'topic modeling', num_topics=topic, Project_name=self.project_name, OriginalScorePercentage=OriginalScorePercentage)
            ei.diagnose()

            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis


        diagnoses = diagnose_real(matrix_name, "normal", None, 0 )
        # original_precision = diagnoses["precision"]
        # original_recall = diagnoses["recall"]
        # original_wasted = diagnoses["wasted"]
        # original_fscore = diagnoses["fscore"]

        row = self._fill_row(diagnoses, matrix_name, 1, False, "original")
        self.rows.append(row)
        rows_combined_methods.append(row)
        # self.rows.append([  "original",
        #                     original_precision,
        #                     original_recall,
        #                     original_wasted,
        #                     self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
        #                     1, #only barinel
        #                     original_fscore,
        #                     diagnoses["expense"],
        #                     diagnoses["tscore"],
        #                     diagnoses["cost"],
        #                     diagnoses["exam_score"],])
        #
        # rows_combined_methods.append(["original",
        #                     original_precision,
        #                     original_recall,
        #                     original_wasted,
        #                     self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
        #                     1, #only barinel
        #                     original_fscore,
        #                     diagnoses["expense"],
        #                     diagnoses["tscore"],
        #                     diagnoses["cost"],
        #                     diagnoses["exam_score"],])



class BarinelTesterOtherAlgorithm(BarinelTester):
    def __init__(self, project_name, technique, local):
        super().__init__(project_name, technique, local)  # represnt what comes out from the github results of other teqniques
        self.technique = technique

    def diagnose(self, matrix_name):
        # getting basic values
        for percentage in self.percentages:
            ei = read_json_planning_file(matrix_name, "CompSimilarity",'other method', Project_name=self.project_name, technique_and_project=f"{self.technique}_{self.project_name}",OriginalScorePercentage=percentage)
            ei.diagnose()
            diagnoses = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics

            row = self._fill_row(diagnoses, matrix_name, percentage, False, self.technique)
            self.rows_all_divisions.append(row)
            # self.rows_all_divisions.append([self.technique,
            #                                 diagnoses["precision"],
            #                                 diagnoses["recall"],
            #                                 diagnoses["wasted"],
            #                                 self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
            #                                 percentage,
            #                                 diagnoses['fscore'],
            #                                 diagnoses["expense"],
            #                                 diagnoses["tscore"],
            #                                 diagnoses["cost"],
            #                                 diagnoses["exam_score"],])
            if percentage == self. optimal_original_score_percentage:
                self.rows.append(row)
                # self.rows.append([self.technique,
                #                   diagnoses["precision"],
                #                   diagnoses["recall"],
                #                   diagnoses["wasted"],
                #                 self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                #                   percentage,
                #                   diagnoses['fscore'],
                #                 diagnoses["expense"],
                #                 diagnoses["tscore"],
                #                 diagnoses["cost"],
                #                 diagnoses["exam_score"],])
                rows_combined_methods.append(row)
                # rows_combined_methods.append([self.technique,
                #                               diagnoses["precision"],
                #                               diagnoses["recall"],
                #                               diagnoses["wasted"],
                #               self.mapping[matrix_name.replace('/', '\\').split('\\')[-1]],
                #                                   percentage, diagnoses['fscore'],
                #                 diagnoses["expense"],
                #                 diagnoses["tscore"],
                #                 diagnoses["cost"],
                #                 diagnoses["exam_score"],])



if __name__ == "__main__":
    project_name = "Codec"
    local = True
    #, 'BLUiR', 'AmaLgam'
    technique = [ "BugLocator", "BRTracer" , 'Locus']
    if len(sys.argv) == 3:
        project_name = sys.argv[1]
        if sys.argv[2] == 'git':
            local = False

    errors = []

    success = []
    failed = []
    topics = list(range(20,601,20))
    all_methods = []
    sanity = BarinelTesterSanity(project_name,local)

    topicModeling = BarinelTesterTopicModeling(project_name, topics, local)
    multiply = BarinelTesterMultiply(project_name, topics, local)
    original = BarinelTesterOriginalMethod(project_name, local)

    all_methods.extend([topicModeling, sanity,original, multiply])

    for t in technique:
        all_methods.append(BarinelTesterOtherAlgorithm(project_name, t, local))

    path = join\
        (str(Path(__file__).parents[1]),'projects',project_name,"barinel","matrixes")

    for matrix in listdir(path):
        try:
            for method in all_methods:
                method.diagnose(join(path, matrix.split("_")[0]))

            success.append(matrix)
        except Exception as e:
            failed.append((matrix, e))
            #print(matrix)
            #raise e
            #errors.append(e)


        print(f"finished a matrix: {matrix}")

    for method in all_methods:
        method.write_rows()



    if local:
        project_path = join(str(Path(__file__).parents[1]),"projects",project_name)
    else:
        project_path = join(str(Path(getcwd())),"projects",project_name)

    experiment2_path = join(project_path,'Experiments', 'Experiment_2')



    path_to_save_into = join(experiment2_path, "data", f"data_all_methods_combined.csv")
    with open(path_to_save_into, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_combined_methods)

    print(f"success:{success},\n failed: {failed}")

