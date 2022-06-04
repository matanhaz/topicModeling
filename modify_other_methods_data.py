from os.path import exists, join, isdir
from os import mkdir,listdir

import json
import csv

import pandas
import tqdm
import sys

from pandas import DataFrame, read_parquet


class ModifyOtherMethods:

    def __init__(self, project_label, group_label, technique, project_folder):
        self.project_label = project_label  # LANG -> like LANG-123
        self.group_label = group_label  # Commons
        self.technique = technique
        self.method_folder_name = '_'.join([technique, project_folder])  # BugLocator_Lang
        self.project_folder = project_folder  # Lang -> like Lang, Weaver etc

        self.project_path = join("projects", project_folder)
        self.analysis_path = join(self.project_path, "analysis")

        self.rows = [
                ['technique', "bug id",
                    "num of files that changed",
                    "num of files that changed no tests",
                    "first index exist files",
                    "max index exist files",
                    "num of files checked exist files",
                    "all indexes",
                    "first index exist files no tests",
                    "max index exist files no tests",
                    "num of files checked exist files no tests",
                    "all indexes no tests",
                    'average similarity'
                ]]

    def change_file_presentation(self):

        dirs = [dir for dir in listdir(
            join("projects", self.project_folder, self.method_folder_name, self.group_label, self.project_label)) if "output" not in dir]

        bug_files = []
        for dir in dirs:
            path = join("projects", self.project_folder, self.method_folder_name, self.group_label, self.project_label, dir, "recommended")
            if exists(path):
                for bug_file in listdir(path):
                    bug_id = bug_file.split('.')[0]
                    bug_files.append((f"{self.project_label}-{bug_id}", join(path,bug_file)))
            else:
                print("not found: "+ path)

        final_dict = {}
        bugs = {}
        for file in bug_files:
            counter = 0
            with open(file[1]) as f:
                bug_name = file[0]
                similarities = []
                for line in f.readlines():
                    parsed_line = line.split("\t")
                    function_name = parsed_line[2].split('.')[-2]
                    if 'test' not in function_name and 'Test' not in function_name:
                        sim =  parsed_line[1] if float(parsed_line[1]) <= 1.0 else '1'
                        similarities.append([function_name, sim, str(counter)])
                        counter += 1
                bugs[bug_name] = similarities

        # saving files results

        final_dict['bugs'] = bugs
        path_to_save = join("projects", self.project_folder,"topicModelingFiles","bug to file and similarity",f"bug_to_file_and_similarity_{self.method_folder_name}")

        data2 = DataFrame.from_dict(final_dict)
        data2.to_parquet(
            path=path_to_save
        )
        # path_to_save = join(self.project_path, "Experiments", "Experiment_1", "data", "methods",self.technique)
        # data2.to_parquet(
        #     path=path_to_save
        # )

        with open(join("projects", self.project_folder,"topicModelingFiles","bug to file and similarity",f"bug_to_file_and_similarity_{self.method_folder_name}.txt"), 'w') as outfile:
            json.dump(final_dict, outfile, indent=4)

        #saving functions results
        final_dict_funcs = {}

        bugs_to_funcs = {}

        with open(join(self.project_path, "analysis", "bug_to_commit_that_solved.txt")) as outfile:
            bugs_to_hex = json.load(outfile)["bugs to commit"]

        df = read_parquet(
            path=join(self.analysis_path, "commitId to all functions")
        )
        commit_to_exist_functions = df.to_dict()["commit id"]

        for bug in bugs:
            bugs_to_funcs[bug] = []
            try:
                commit_hex = [b['hexsha'] for b in bugs_to_hex if b['bug id'] == bug][0]
            except:
                print(bug)
                continue
            commit_id = [commit for commit in commit_to_exist_functions if commit_to_exist_functions[commit]['hexsha'] == commit_hex][0]

            exist_files = commit_to_exist_functions[commit_id]['file to functions']
            exist_files_filtered = {}
            for f in exist_files:
                if exist_files[f] is not None:
                    exist_files_filtered[f.split('\\')[-1].split('/')[-1]] = exist_files[f].tolist()

            counter = 0
            for file in bugs[bug]:
                file_name = file[0] + '.java'
                sim = file[1] if float(file[1]) <1 else '1'
                try:
                    for func in exist_files_filtered[file_name]:
                        bugs_to_funcs[bug].append([func,sim, str(counter), file[0] ])
                        counter += 1
                except:
                    print()
        final_dict_funcs['bugs'] = bugs_to_funcs
        path_to_save = join("projects", self.project_folder,"topicModeling","bug to functions and similarity",f"bug_to_function_and_similarity_{self.method_folder_name}")

        data2 = DataFrame.from_dict(final_dict_funcs)
        data2.to_parquet(
            path=path_to_save
        )
        print()

    # write final dict to a json file


    def gather_otherMethod_topK(self):
        '''first i need to check which bug exist in my project and their project
            then i need to check which functions has benn changed in those bugs
            then i calculate the top k '''

        with open(join("projects",self.project_folder,"analysis","bug_to_commit_that_solved.txt"), 'r') as outfile:
            #with open(join("projects",self.project_folder,"analysis","commitId to all functions.txt"), 'r') as f:
            df = pandas.read_parquet(path=join(self.analysis_path,"commitId to all functions"))
            commit_to_exist_functions =df.to_dict()['commit id']
            all_bugs = json.load(outfile)['bugs to commit']
            #commit_to_exist_functions = json.load(f)['commit id']
            exist_bugs_and_changed_functions = {}
            for bug in all_bugs:
                functions_that_changed = [func.split('\\')[-1].split('/')[-1] for func in bug["files that changed"]]

                functions_that_changed_no_tests = [func.split('\\')[-1].split('/')[-1] for func in bug["files that changed"] if ('test' not in func and 'Test' not in func)]

                commit_id = [commit for commit in commit_to_exist_functions if commit_to_exist_functions[commit]['hexsha'] == bug['hexsha']][0]

                exists_functions = [func.split('\\')[-1].split('/')[-1] for func in list(commit_to_exist_functions[commit_id]['file to functions'].keys())]

                exists_functions_no_tests = [func for func in exists_functions if ('test' not in func and 'Test' not in func)]

                exist_bugs_and_changed_functions[bug['bug id']] = {'function that changed':functions_that_changed,
                                                                   'function that changed no tests':functions_that_changed_no_tests,
                                                                   'exists functions': exists_functions,
                                                                   'exists functions no tests': exists_functions_no_tests}

        with open(join("projects", self.project_folder,"topicModelingFiles","bug to file and similarity",f"bug_to_file_and_similarity_{self.method_folder_name}.txt")) as outfile:
            all_bugs = json.load(outfile)['bugs']
            for bug in all_bugs:
                for file_and_sim in all_bugs[bug]:
                    file_and_sim[0] += '.java'
            exists_bugs = exist_bugs_and_changed_functions.keys()
            bug_to_miss = {}
            for bug in tqdm.tqdm(all_bugs):
                if bug not in exists_bugs:
                    continue

                functions_that_changed = exist_bugs_and_changed_functions[bug]['function that changed']
                exists_functions = exist_bugs_and_changed_functions[bug]['exists functions']
                min_index, max_index, num_functions, all_indexes, tmp, average_sim = \
                    self.find_indexes_exist_functions(functions_that_changed, all_bugs[bug], exists_functions)

                functions_that_changed_no_tests = exist_bugs_and_changed_functions[bug]['function that changed no tests']
                exists_functions_no_tests = exist_bugs_and_changed_functions[bug]['exists functions no tests']
                min_index_no_tests, max_index_no_tests, num_functions_no_tests, all_indexes_no_tests, miss, average_sim_no_test = \
                    self.find_indexes_exist_functions(functions_that_changed_no_tests, all_bugs[bug], exists_functions_no_tests)

                bug_to_miss[bug] = miss

                self.rows.append([self.technique,bug,len(functions_that_changed),len(functions_that_changed_no_tests),
                                  min_index,max_index,num_functions,all_indexes,
                                 min_index_no_tests,max_index_no_tests,num_functions_no_tests,all_indexes_no_tests,average_sim_no_test ])

            with open(join(self.project_path, "Experiments", "Experiment_1", "data", f"{self.method_folder_name}_indexes.csv"), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.rows)

    def find_indexes_exist_functions(self,changed_functions,  funcs_and_similarities, exists_functions):
        max_index = -1

        exist_funcs_with_similarity = []

        missing_functions = []

        for func_exist in set(exists_functions):
            for func_and_similarity in funcs_and_similarities:
                if func_exist == func_and_similarity[0]:
                    exist_funcs_with_similarity.append(func_and_similarity)
                    break

        exist_funcs_with_similarity.sort(key=lambda x: x[1], reverse=True)

        if len(changed_functions) == 0:
            return -1,-1, len(exist_funcs_with_similarity),[], None, 0

        min_index = len(exist_funcs_with_similarity)
        all_indexes = []
        average_similarity = 0.0
        for func in changed_functions:
            index = 0
            for exist_func_and_similarity in exist_funcs_with_similarity:
                if func == exist_func_and_similarity[0]:
                    max_index = max(max_index, index)
                    min_index = min(min_index, index)
                    all_indexes.append(index)
                    average_similarity += exist_func_and_similarity[1]
                    break
                index += 1
            else:
                max_index = max(max_index, index)
                min_index = min(min_index, index)
                all_indexes.append(index)
                missing_functions.append(func)

        average_similarity /= len(changed_functions)
        return min_index, max_index, len(exist_funcs_with_similarity), all_indexes, missing_functions, average_similarity



if __name__ == "__main__":
    # two arguments :
    # 1. project
    # 2. technique



    if len(sys.argv) == 2:
        selected_project = sys.argv[1]
    else:
        selected_project = "Codec"

    with open("project_info.txt", 'r') as outfile:
        data = json.load(outfile)

        project = data[selected_project]['project']
        group = data[selected_project]['group']

    for dir in listdir(join("projects", selected_project)):
        if isdir(join("projects", selected_project,dir)) and selected_project in dir:
            modify = ModifyOtherMethods(project, group, dir.split('_')[0], selected_project)
            modify.change_file_presentation()
            modify.gather_otherMethod_topK()

