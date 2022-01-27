from os.path import exists, join, isdir
from os import mkdir,listdir

import json
import csv

import pandas
import tqdm
import sys

from pandas import DataFrame


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
                ['technique', 'bug id',
                 'num of functions that changed no tests' ,
                 'max index exist functions no tests',
                 'num of functions checked']]

    def change_file_presentation(self):

        dirs = [dir for dir in listdir(
            join("projects", self.project_folder, self.method_folder_name, self.group_label, self.project_label)) if "output" not in dir]

        bug_files = []
        for dir in dirs:
            path = join("projects", self.project_folder, self.method_folder_name, self.group_label, self.project_label, dir, "recommended")
            for bug_file in listdir(path):
                bug_id = bug_file.split('.')[0]
                bug_files.append((f"{self.project_label}-{bug_id}", join(path,bug_file)))

        final_dict = {}
        bugs = {}
        for file in bug_files:
            counter = 0
            with open(file[1]) as f:
                bug_name = file[0]
                sim = []
                for line in f.readlines():
                    parsed_line = line.split("\t")
                    function_name = parsed_line[2].split('.')[-2]
                    if 'test' not in function_name and 'Test' not in function_name:
                        sim.append([function_name, parsed_line[1], str(counter)])
                        counter += 1
                bugs[bug_name] = sim


        final_dict['bugs'] = bugs
        path_to_save = join("projects", self.project_folder,"topicModeling","bug to funcion and similarity",f"bug_to_function_and_similarity_{self.method_folder_name}")

        data2 = DataFrame.from_dict(final_dict)
        data2.to_parquet(
            path=path_to_save
        )

        with open(join("projects", self.project_folder,"topicModeling","bug to funcion and similarity",f"bug_to_function_and_similarity_{self.method_folder_name}.txt"), 'w') as outfile:
            json.dump(final_dict, outfile, indent=4)

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
                functions_that_changed = [func['function name'] for func in bug["function that changed"]]

                functions_that_changed_no_tests = [func for func in functions_that_changed if not ('test' in func or 'Test' in func)]

                exists_functions = [commit_to_exist_functions[commit]['all functions'].tolist() for commit in commit_to_exist_functions if commit_to_exist_functions[commit]['hexsha'] == bug['hexsha']][0]

                exists_functions_no_tests = [func for func in exists_functions if not ('test' in func or 'Test' in func)]

                exist_bugs_and_changed_functions[bug['bug id']] = {'function that changed':functions_that_changed,
                                                                   'function that changed no tests':functions_that_changed_no_tests,
                                                                   'exists functions': exists_functions,
                                                                   'exists functions no tests': exists_functions_no_tests}

        with open(join("projects", self.project_folder,"topicModeling","bug to funcion and similarity",f"bug_to_function_and_similarity_{self.method_folder_name}.txt")) as outfile:
            all_bugs = json.load(outfile)['bugs']
            exists_bugs = exist_bugs_and_changed_functions.keys()
            for bug in tqdm.tqdm(all_bugs):
                if bug not in exists_bugs:
                    continue

     #          max_index = -1 if len(exist_bugs_and_changed_functions[bug]['function that changed']) == 0 else max([[function[2] for function in all_bugs[bug] if function[0] == func][0] for func in exist_bugs_and_changed_functions[bug]['function that changed']])
                all_indexes = [[function[2] for function in all_bugs[bug] if function[0] == func] for func in exist_bugs_and_changed_functions[bug]['function that changed no tests']]
                max_index_no_test = -1 if len(all_indexes)==0 else len(exist_bugs_and_changed_functions[bug]['exists functions no tests'])-1 if [] in all_indexes else max(all_indexes)[0]
                self.rows.append([self.technique,bug,
                             len(exist_bugs_and_changed_functions[bug]['function that changed no tests']),
                             max_index_no_test,
                             len(exist_bugs_and_changed_functions[bug]['exists functions no tests'])])

            with open(join(self.project_path, "Experiments", "Experiment_1", "data", f"{self.method_folder_name}_indexes.csv"), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.rows)


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

