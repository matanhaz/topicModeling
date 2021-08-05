

def run_dignose_eyal(bug_id, results_file, additional_files_path):
    from .Diagnoser.diagnoserUtils import readPlanningFile
    from .Diagnoser.Diagnosis_Results import Diagnosis_Results
    from operator import itemgetter
    import sfl.Diagnoser.Experiment_Data
    import sfl.Diagnoser.ExperimentInstance
    import sfl.Planner.HP_Random
    import timeit
    import time
    import os
    print("Start diagnose")
    start = time.time()
    os.remove(results_file)
    print("Starting oracle diagnosis")
    list_pre_oracle, list_recall_oracle = get_diagnose_results(1, os.path.join(additional_files_path,
                                                                               r'inputMatrix_oracle.txt'))
    print("Starting predicted diagnosis")
    list_pre_prob, list_recall_prob = get_diagnose_results(2, os.path.join(additional_files_path,
                                                                           r'inputMatrix_prediction.txt'))
    print("Starting baseline diagnosis")
    list_pre_baseline, list_recall_baseline = get_diagnose_results(1, os.path.join(additional_files_path,
                                                                                   r'inputMatrix_baseline.txt'))
    print("Starting random diagnosis")
    list_pre_random, list_recall_random = get_diagnose_results(3, os.path.join(additional_files_path,
                                                                               r'inputMatrix_oracle.txt'))

    if len(list_pre_oracle) > 0:
        write_result_to_file(results_file, bug_id, "oracle", list_pre_oracle, list_recall_oracle)
        write_result_to_file(results_file, bug_id, "prediction", list_pre_prob, list_recall_prob)
        write_result_to_file(results_file, bug_id, "baseline", list_pre_baseline, list_recall_baseline)
        write_result_to_file(results_file, bug_id, "random", list_pre_random, list_recall_random)
    total_diagnose_time = time.time() - start
    print("Total diagnose time: " + str(total_diagnose_time))
    i = 5


def write_result_to_file(RESULTS_FILE, bug_id, name, list_pre, list_recall):
    with open(RESULTS_FILE, 'a') as fd:
        str_pre = ','.join([str(x) for x in list_pre])
        fd.write(str(name) + ":" + "precision" + ":" + str(bug_id) + "," + str_pre + "\n")
        str_recall = ','.join([str(x) for x in list_recall])
        fd.write(str(name) + ":" + "recall" + ":" + str(bug_id) + "," + str_recall + "\n")



def get_diagnose_results(param, file_path):
    instAmir = readPlanningFile(file_path)
    instAmir.diagnose()
    list_pre, list_recall = sfl.Planner.HP_Random.main_HP(instAmir, param)
    return list_pre, list_recall


if __name__ == '__main__':
    print("start init diagnoser")
    start_t = timeit.default_timer()
    run_dignose_eyal("2", r'C:\Users\eyalhad\Desktop\runningProjects\Lang_version\results3.csv',
                     r'C:\Users\eyalhad\Desktop\runningProjects\Math_version\math_2_fix\additionalFiles')
    total_t = timeit.default_timer() - start_t
    print("get_diagnose_results: " + str(total_t / 60) + "\r\n")
