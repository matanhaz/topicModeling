import csv
import json
from os.path import join, exists, isdir, isfile, islink
from os import mkdir, listdir, rename, unlink
import shutil
from pathlib import Path

import pandas as pd

from sfl.Diagnoser.diagnoserUtils import (
    read_json_planning_file,
    read_json_planning_instance,
    write_json_planning_file,
)
from sfl.Diagnoser.Diagnosis_Results import Diagnosis_Results
from sfl.Diagnoser.Experiment_Data import Experiment_Data


class BarinelTester:
    def __init__(self, project_name, test_type):
        self.project_name = project_name
        self.epsilon = 0.01
        self.rows = []
        self.test_type = test_type
        self.project_path = join("..","projects",project_name)
        self.experiment2_path = join(self.project_path,'Experiments', 'Experiment_2')
        self.prepare_dir()
        self.prepare_matrixes()

    def prepare_dir(self):
        if not exists(join(self.project_path, "barinel")):
            mkdir(join(self.project_path, "barinel"))

        # generate matrixes somehow, need amir's code

    def prepare_matrixes(self):
        with open(join(self.project_path, "analysis", "bug_to_commit_that_solved.txt")) as f:
            data = json.loads(f.read())["bugs to commit"]  # array of dicts, each represent a bug that i discovered

        old_path_matrixes = join\
            (self.project_path, "barinel", "matrixes_before_change")
        new_path_matrixes = join\
            (self.project_path, "barinel", "matrixes")

        """move avtive-bugs.csv"""
        rename(join(old_path_matrixes,'active-bugs.csv'),join(self.project_path, "barinel"))


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
        for m in all_matrixes_in_dir:
            for bug in data:
                if bug["bug id"].split("-")[1] == m[1]:
                    m.append(bug["hexsha"])
                    break

        """treansfer matrixes to a new location"""
        counter = 1
        for m in all_matrixes_in_dir:
            if isfile(join
                                  (new_path_matrixes, m[2])):
                rename(join
                          (new_path_matrixes, m[0]), join
                (new_path_matrixes, f"{m[2]}_{str(counter)}"))
                counter += 1
            else:
                rename(join
                          (new_path_matrixes, m[0]), join
                (new_path_matrixes, m[2]))

    def write_rows(self):
        if not exists(self.experiment2_path):
            mkdir(self.experiment2_path)
            mkdir(join(self.experiment2_path, "data"))


        path_to_save_into = join(self.experiment2_path, "data", f"data_{self.test_type}.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
            pass


class BarinelTesterSanity(BarinelTester):
    def __init__(self, project_name):
        super().__init__(project_name, "Sanity")
        self.high_similarity = [0.6, 0.7, 0.8, 0.9, 1]
        self.low_similarity = [0.4, 0.3, 0.2, 0.1]
        self.rows.append(
            [
                "real bug diagnose sim",
                "unreal bug diagnose sim",
                "original precision",
                "original recall",
                "original wasted",
                "precision-comp",
                "recall-comp",
                "wasted-comp",
                "precision-tests",
                "recall-tests",
                "wasted-tests",
                "precision-both",
                "recall-both",
                "wasted-both",
            ]
        )

    def diagnose(self, matrix_name):
        # getting basic values

        def diagnose_sanity(matrix_name, exp_type, good_sim, bad_sim):
            ei = read_json_planning_file(matrix_name,exp_type,'sanity', good_sim=good_sim, bad_sim=bad_sim)
            ei.diagnose()
            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis

        diagnoses = diagnose_sanity(matrix_name, "normal", 0, 0)
        original_precision = diagnoses["precision"]
        original_recall = diagnoses["recall"]
        original_wasted = diagnoses["wasted"]

        if original_precision < self.epsilon or original_precision > 1 - self.epsilon:
            return

        for good_sim in self.high_similarity:
            for bad_sim in self.low_similarity:
                diagnosis_comp = diagnose_sanity(matrix_name, "CompSimilarity", good_sim, bad_sim)
                diagnosis_tests = diagnose_sanity(matrix_name, "TestsSimilarity", good_sim, bad_sim)
                diagnosis_both = diagnose_sanity(matrix_name, "BothSimilarities", good_sim, bad_sim)

                self.rows.append(
                    [
                        good_sim,
                        bad_sim,
                        original_precision,
                        original_recall,
                        original_wasted,
                        diagnosis_comp["precision"],
                        diagnosis_comp["recall"],
                        diagnosis_comp["wasted"],
                        diagnosis_tests["precision"],
                        diagnosis_tests["recall"],
                        diagnosis_tests["wasted"],
                        diagnosis_both["precision"],
                        diagnosis_both["recall"],
                        diagnosis_both["wasted"],
                    ]
                )


class BarinelTesterTopicModeling(BarinelTester):
    def __init__(self, project_name, topics_range):
        super().__init__(project_name, "TopicModeling")
        self.topics = range(topics_range[0], topics_range[1])
        self.rows.append(
            [
                "num topics",
                "original precision",
                "original recall",
                "original wasted",
                "precision-comp",
                "recall-comp",
                "wasted-comp",
                "precision-tests",
                "recall-tests",
                "wasted-tests",
                "precision-both",
                "recall-both",
                "wasted-both",
            ]
        )

    def diagnose(self, matrix_name):
        # getting basic values

        def diagnose_real(matrix_name, exp_type, topic):
            ei = read_json_planning_file(matrix_name, exp_type,'topic modeling', num_topics=topic, Project_name=self.project_name)
            ei.diagnose()
            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis

       
        diagnoses = diagnose_real(matrix_name, "normal", None)
        original_precision = diagnoses["precision"]
        original_recall = diagnoses["recall"]
        original_wasted = diagnoses["wasted"]

        if original_precision < self.epsilon or original_precision > 1 - self.epsilon:
            return

        for topic in self.topics:
            diagnosis_comp = diagnose_real(matrix_name, "CompSimilarity", topic)
            diagnosis_tests = diagnose_real(matrix_name, "TestsSimilarity", topic)
            diagnosis_both = diagnose_real(matrix_name, "BothSimilarities", topic)
            print('finished topics ', topic)
            self.rows.append(
                [
                    topic,
                    original_precision,
                    original_recall,
                    original_wasted,
                    diagnosis_comp["precision"],
                    diagnosis_comp["recall"],
                    diagnosis_comp["wasted"],
                    diagnosis_tests["precision"],
                    diagnosis_tests["recall"],
                    diagnosis_tests["wasted"],
                    diagnosis_both["precision"],                    diagnosis_both["recall"],
                    diagnosis_both["wasted"],
                ]
            )


class BarinelTesterOtherAlgorithm(BarinelTester):
    def __init__(self, project_name, technique):
        super().__init__(project_name, technique)  # represnt what comes out from the github results of other teqniques
        self.rows.append(["technique","precision", "recall", "wasted"])
        self.technique = technique

    def diagnose(self, matrix_name):
        # getting basic values
        ei = read_json_planning_file(matrix_name, "CompSimilarity",'other method', Project_name=self.project_name, technique_and_project=self.technique)
        ei.diagnose()
        diagnoses = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics

        precision = diagnoses["precision"]
        recall = diagnoses["recall"]
        wasted = diagnoses["wasted"]

        self.rows.append([self.technique, diagnoses["precision"], diagnoses["recall"], diagnoses["wasted"]])

import sys
if __name__ == "__main__":

    project_name = "apache_commons-lang"
    technique = "BugLocator"
    if len(sys.argv) != 3:
        print("missing arguments")
        exit()


    project_name = sys.argv[1]
    technique = sys.argv[2]
    success = []
    failed = []

    # select the test we want to do
    sanity = BarinelTesterSanity(project_name)
    topicModeling = BarinelTesterTopicModeling(project_name, (15, 26))
    other_method = BarinelTesterOtherAlgorithm(project_name, f"{technique}_{project_name}")

    path = join\
        (str(Path(__file__).parents[1]),'projects',project_name,"barinel","matrixes")

    for matrix in listdir(path):
        try:
            sanity.diagnose(join(path,"matrixes", matrix.split("_")[0]))
            topicModeling.diagnose(join
                                   (path, matrix.split("_")[0]))
            other_method.diagnose(join(path,"matrixes", matrix.split("_")[0]))
            success.append(matrix)
        except Exception as e:
            #raise e

            failed.append(matrix)
        print(f"finished a matrix: {matrix}")

    sanity.write_rows()
    topicModeling.write_rows()
    other_method.write_rows()

    print(f"success:{success},\n failed: {failed}")
