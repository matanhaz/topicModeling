import csv
import json
import os

import pandas as pd

# from tqdm import tqdm


from sfl.Diagnoser.diagnoserUtils import (
    read_json_planning_file,
    read_json_planning_file_real,
    read_json_planning_instance,
    write_json_planning_file,
)
from sfl.Diagnoser.Diagnosis_Results import Diagnosis_Results
from sfl.Diagnoser.Experiment_Data import Experiment_Data

PROJECT_NAME = "apache_commons-lang"
TOPICS = range(15, 26)
HIGHER_SIMILARITIES = [0.6, 0.7, 0.8, 0.9, 1]
LOWER_SIMILARITIES = [0.4, 0.3, 0.2, 0.1]
EPSILON = 0.01

rows_fake = []
rows_fake.append(
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
rows_real = []
rows_real.append(
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
other_rows = ["precision", "recall", "wasted"]


with open(
    f"..\\..\\projects\\{PROJECT_NAME}\\analysis\\bug_to_commit_that_solved.txt"
) as f:
    data = json.loads(f.read())[
        "bugs to commit"
    ]  # array of dicts, each represent a bug that i discovered


"""remove json"""
for file in os.listdir(os.getcwd() + "\\matrixes_before_change"):
    new_name = file.split(".")[0]
    os.rename(f"matrixes_before_change\\{file}", f"matrixes_before_change\\{new_name}")

"""read csv"""

df = pd.read_csv("active-bugs.csv").to_dict()
all_matrixes_in_dir = []
for i in range(len(df["bug.id"])):
    if str(df["bug.id"][i]) in os.listdir(os.getcwd() + "\\matrixes_before_change"):
        all_matrixes_in_dir.append(  # (bug file index, bug actual id in jira)
            [df["bug.id"][i], df["report.id"][i].split("-")[1]]
        )

"""add to each matrix his hexsha"""
for m in all_matrixes_in_dir:
    for bug in data:
        if bug["bug id"].split("-")[1] == m[1]:
            m.append(bug["hexsha"])
            break

"""treansfer matrixes to a new location"""
counter = 1
for m in all_matrixes_in_dir:
    if os.path.isfile(f"matrixes\\{m[2]}"):
        os.rename(f"matrixes_before_change\\{m[0]}", f"matrixes\\{m[2]}_{str(counter)}")
        counter += 1
    else:
        os.rename(f"matrixes_before_change\\{m[0]}", f"matrixes\\{m[2]}")


def diagnose_fake_similarities(matrix_name):
    # getting basic values

    def diagnose_fake(matrix_name, exp_type, good_sim, bad_sim):
        ei = read_json_planning_file(matrix_name, good_sim, bad_sim, exp_type)
        ei.diagnose()
        diagnosis = Diagnosis_Results(
            ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()
        ).metrics
        return diagnosis

    diagnoses = diagnose_fake(matrix_name, 0, 0, "normal")
    original_precision = diagnoses["precision"]
    original_recall = diagnoses["recall"]
    original_wasted = diagnoses["wasted"]

    if original_precision < EPSILON or original_precision > 1 - EPSILON:
        return

    for good_sim in HIGHER_SIMILARITIES:
        for bad_sim in LOWER_SIMILARITIES:
            diagnosis_comp = diagnose_fake(
                matrix_name, good_sim, bad_sim, "CompSimilarity"
            )
            diagnosis_tests = diagnose_fake(
                matrix_name, good_sim, bad_sim, "TestsSimilarity"
            )
            diagnosis_both = diagnose_fake(
                matrix_name, good_sim, bad_sim, "BothSimilarities"
            )

            # print("e1 = %f , e2 = %f, e3 = %f" %(CompSimilarity[1],CompSimilarity[2],CompSimilarity[3]))
            # print(Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics)
            # print(Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).diagnoses)
            # print(diagnosis_comp['precision'])
            rows_fake.append(
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


def diagnose_real_similarities(matrix_name):
    # getting basic values

    def diagnose_real(matrix_name, exp_type, topic):
        ei = read_json_planning_file_real(matrix_name, exp_type, topic, PROJECT_NAME)
        ei.diagnose()
        diagnosis = Diagnosis_Results(
            ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()
        ).metrics
        return diagnosis

    ei = read_json_planning_file(matrix_name, 0, 0, experiment_type="normal")
    ei.diagnose()
    diagnoses = Diagnosis_Results(
        ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()
    ).metrics
    original_precision = diagnoses["precision"]
    original_recall = diagnoses["recall"]
    original_wasted = diagnoses["wasted"]

    if original_precision < EPSILON or original_precision > 1 - EPSILON:
        return

    for topic in TOPICS:
        diagnosis_comp = diagnose_real(matrix_name, "CompSimilarity", topic)
        diagnosis_tests = diagnose_real(matrix_name, "TestsSimilarity", topic)
        diagnosis_both = diagnose_real(matrix_name, "BothSimilarities", topic)

        rows_real.append(
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


def diagnose_other_methods(matrix_name):
    # getting basic values
    ei = read_json_planning_file_real(
        matrix_name, "CompSimilarity", 0, PROJECT_NAME, True, "BugLocator_Lang"
    )
    ei.diagnose()
    diagnoses = Diagnosis_Results(
        ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()
    ).metrics

    precision = diagnoses["precision"]
    recall = diagnoses["recall"]
    wasted = diagnoses["wasted"]

    other_rows.append(
        [diagnoses["precision"], diagnoses["recall"], diagnoses["wasted"]]
    )


def write_rows(rows, real_fake):
    with open("data_" + real_fake + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        pass


testing_topic_modeling = False

success = []
failed = []

with open(
    os.path.dirname(__file__)
    + f"\\..\\..\\projects\\{PROJECT_NAME}\\topicModeling\\bug to funcion and similarity\\bug_to_function_and_similarity_BugLocator_Lang.txt"
) as f:
    d = json.loads(f.read())["bugs"]
    # for matrix in os.listdir("matrixes"):
    for matrix in all_matrixes_in_dir:
        bug_id = list(
            x["bug id"] for x in data if x["hexsha"] == matrix[2].split("_")[0]
        )[0]
        # if testing_topic_modeling or bug_id in d.keys():
        try:
            with open("matrixes//" + matrix[2].split("_")[0], "r") as f:
                t_counter = 0
                instance = json.loads(f.read())
                for t in instance["tests_details"]:
                    if t[2] == 1:
                        t_counter += 1
            diagnose_real_similarities("matrixes//" + matrix[2].split("_")[0])
            success.append(matrix[0])
        except:
            failed.append(matrix[0])
        write_rows(rows_real, "real")
        print(f"success:{success},\n failed: {failed}")
