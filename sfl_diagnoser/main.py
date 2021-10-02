import csv
import json
import os
import shutil
import pandas as pd

from sfl.Diagnoser.diagnoserUtils import (
    read_json_planning_file,
    read_json_planning_file_real,
    read_json_planning_instance,
    write_json_planning_file,
)
from sfl.Diagnoser.Diagnosis_Results import Diagnosis_Results
from sfl.Diagnoser.Experiment_Data import Experiment_Data


class BarinelTester:
    def __init__(self, project_name, test_type):
        self.project_name = project_name
        self.epsilin = 0.01
        self.rows = []
        self.test_type = test_type
        self.project_path = f"..\\projects\\{project_name}"
        self.prepare_dir()
        self.prepare_matrixes()

    def prepare_dir(self):
        if os.path.exists(os.path.join(self.project_path, "barinel")):
            os.mkdir(os.path.join(self.project_path, "barinel"))

        # generate matrixes somehow, need amir's code

    def prepare_matrixes(self):
        with open(os.path.join(self.project_path, "analysis", "bug_to_commit_that_solved.txt")) as f:
            data = json.loads(f.read())["bugs to commit"]  # array of dicts, each represent a bug that i discovered

        old_path_matrixes = os.path.join(self.project_path, "barine", "matrixes_before_change")
        new_path_matrixes = os.path.join(self.project_path, "barinel", "matrixes")

        """clear matrixes dir"""
        for file in os.listdir(new_path_matrixes):
            file_path = os.path.join(new_path_matrixes, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

        """copy files"""
        for file in os.listdir(old_path_matrixes):
            shutil.copy(
                os.path.join(old_path_matrixes, file),
                os.path.join(new_path_matrixes, file),
            )

        """remove json"""
        for file in os.listdir(new_path_matrixes):
            new_name = file.split(".")[0]
            os.rename(os.path.join(new_path_matrixes, file), os.path.join(new_path_matrixes, new_name))

        """read csv"""
        df = pd.read_csv("active-bugs.csv").to_dict()
        all_matrixes_in_dir = []
        for i in range(len(df["bug.id"])):
            if str(df["bug.id"][i]) in os.listdir(new_path_matrixes):
                all_matrixes_in_dir.append([df["bug.id"][i], df["report.id"][i].split("-")[1]])  # (bug file index, bug actual id in jira)

        """add to each matrix his hexsha"""
        for m in all_matrixes_in_dir:
            for bug in data:
                if bug["bug id"].split("-")[1] == m[1]:
                    m.append(bug["hexsha"])
                    break

        """treansfer matrixes to a new location"""
        counter = 1
        for m in all_matrixes_in_dir:
            if os.path.isfile(os.path.join(new_path_matrixes, m[2])):
                os.rename(os.path.join(new_path_matrixes, m[0]), os.path.join(new_path_matrixes, f"{m[2]}_{str(counter)}"))
                counter += 1
            else:
                os.rename(os.path.join(new_path_matrixes, m[0]), os.path.join(new_path_matrixes, m[2]))

    def write_rows(self):
        path_to_save_into = os.path.join(self.project_path, "barinel", f"data_{self.test_type}.csv")
        with open(path_to_save_into, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
            pass


class BarinelTesterSanity(BarinelTester):
    def __init__(self, project_name):
        super().__init__(self, project_name, "Sanity")
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
            ei = read_json_planning_file(matrix_name, good_sim, bad_sim, exp_type)
            ei.diagnose()
            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis

        diagnoses = diagnose_sanity(matrix_name, 0, 0, "normal")
        original_precision = diagnoses["precision"]
        original_recall = diagnoses["recall"]
        original_wasted = diagnoses["wasted"]

        if original_precision < self.epsilon or original_precision > 1 - self.epsilon:
            return

        for good_sim in self.high_similarity:
            for bad_sim in self.low_similarity:
                diagnosis_comp = diagnose_sanity(matrix_name, good_sim, bad_sim, "CompSimilarity")
                diagnosis_tests = diagnose_sanity(matrix_name, good_sim, bad_sim, "TestsSimilarity")
                diagnosis_both = diagnose_sanity(matrix_name, good_sim, bad_sim, "BothSimilarities")

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
        super().__init__(self, project_name, "TopicModeling")
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
            ei = read_json_planning_file_real(matrix_name, exp_type, topic, self.project_name)
            ei.diagnose()
            diagnosis = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
            return diagnosis

        ei = read_json_planning_file(matrix_name, 0, 0, experiment_type="normal")
        ei.diagnose()
        diagnoses = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
        original_precision = diagnoses["precision"]
        original_recall = diagnoses["recall"]
        original_wasted = diagnoses["wasted"]

        if original_precision < self.epsilon or original_precision > 1 - self.epsilon:
            return

        for topic in self.topics:
            diagnosis_comp = diagnose_real(matrix_name, "CompSimilarity", topic)
            diagnosis_tests = diagnose_real(matrix_name, "TestsSimilarity", topic)
            diagnosis_both = diagnose_real(matrix_name, "BothSimilarities", topic)

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
                    diagnosis_both["precision"],
                    diagnosis_both["recall"],
                    diagnosis_both["wasted"],
                ]
            )


class BarinelTesterOtherAlgorithm(BarinelTester):
    def __init__(self, project_name, technique):
        super().__init__(self, project_name, technique)  # represnt what comes out from the github results of other teqniques
        self.rows.append(["precision", "recall", "wasted"])

    def diagnose(self, matrix_name):
        # getting basic values
        ei = read_json_planning_file_real(matrix_name, "CompSimilarity", 0, self.project_name, True, self.technique)
        ei.diagnose()
        diagnoses = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics

        precision = diagnoses["precision"]
        recall = diagnoses["recall"]
        wasted = diagnoses["wasted"]

        self.rows.append([diagnoses["precision"], diagnoses["recall"], diagnoses["wasted"]])

import sys
if __name__ == "__main__":

    project_name = "Lang"
    if len(sys.argv) != 3:
        print("missing arguments")


    project_name = sys.argv[2]
    technique = sys.argv[3]
    success = []
    failed = []

    # select the test we want to do
    sanity = BarinelTesterSanity(project_name)
    topicModeling = BarinelTesterTopicModeling(project_name, (15, 26))
    buglocator_lang = BarinelTesterOtherAlgorithm(project_name, f"{technique}_{project_name}")

    path = os.path.join("..","projects",project_name,"barinel")

    for matrix in os.listdir(os.path.join(path,"matrixes")):
        try:
            sanity.diagnose(os.path.join(path,"matrixes", matrix.split("_")[0]))
            topicModeling.diagnose(os.path.join(path,"matrixes", matrix.split("_")[0]))
            buglocator_lang.diagnose(os.path.join(path,"matrixes", matrix.split("_")[0]))
            success.append(matrix)
        except Exception as e:
            # raise e
            failed.append(matrix)
        print(f"finished a matrix: {matrix}")

        sanity.write_rows()
        topicModeling.write_rows()
        buglocator_lang.write_rows()

    print(f"success:{success},\n failed: {failed}")
