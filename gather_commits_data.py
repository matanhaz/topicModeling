from os.path import exists, join
from os import mkdir
import json
import sys
from pydriller import repository

from pandas import *


class GatherCommitsData:
    def __init__(self, git_url, project_name):
        self.project_path = join("projects", project_name)
        self.data_path = join(self.project_path, "data")
        self.analysis_path = join(self.project_path, "analysis")

        self.git = git_url
        self.changes_to_commits = []
        self.functions = []
        self.func_name_to_params_and_commits = {}
        self.existing_functions = []
        self.commit_id_to_functions = {}

        self.commit_index = 1

        self.create_dirs()



    def gather(self):
        commits = repository.Repository(self.git, num_workers=4, only_modifications_with_file_types=['.java']).traverse_commits()

        for commit in commits:
            if not commit.in_main_branch:
                self.commit_index += 1
                continue

            # data0, data1 ....
            modified_functions = self.extract_modified_functions(commit.modified_files)
            self.gather_commit_changes(commit, modified_functions)

            # func to commits
            self.gather_func_to_commits(commit, modified_functions)

            # commit id to all functions
            self.gather_all_functions(commit, commit.modified_files)

            self.commit_index += 1

        file_number = self.get_file_number()

        for func in self.func_name_to_params_and_commits.keys():
            self.functions.append({
                'function name' : func,
                'function params': self.func_name_to_params_and_commits[func]['params'],
                'commits that changed in': self.func_name_to_params_and_commits[func]['commits that changed in']
            })

        function_to_commits_dic = {}
        function_to_commits_dic['functions'] = self.functions
        with open(join(self.data_path, "funcToCommits.txt"), 'w') as outfile:
            json.dump(function_to_commits_dic, outfile, indent=4)

        self.save_into_file("data"+str(file_number), {'changes':self.changes_to_commits}, 'changes')

        # takes time, maybe seperate to 2 files

        # index = 0
        # func_name_to_index = {}
        # for commit in self.commit_id_to_functions:
        #     for i,func in enumerate(self.commit_id_to_functions[commit]['all functions']):
        #         if func not in func_name_to_index:
        #             func_name_to_index[func] = index
        #             index += 1
        #         self.commit_id_to_functions[commit]['all functions'][i] = func_name_to_index[func]
        #
        #
        #
        #
        # self.save_functions_per_commit('function to func id', func_name_to_index, 'func name')


        self.save_functions_per_commit('commitId to all functions', self.commit_id_to_functions, 'commit id')

    def create_dirs(self):
        if not (exists(self.project_path)):
            mkdir(self.project_path)
        if not exists(self.analysis_path):
            mkdir(self.analysis_path)
        if not exists(self.data_path):
            mkdir(self.data_path)

    def get_file_number(self):
        return int(self.commit_index / 1000)

    def gather_commit_changes(self,commit, list_of_modified_functions):
        file_number = self.get_file_number()

        new_change = {
            'commit_id': self.commit_index,
            'commit_summary': commit.msg,
            'functions': list_of_modified_functions
        }
        self.changes_to_commits.append(new_change)

        if self.commit_index % 250 == 0:
            self.save_into_file("data"+str(file_number), {'changes':self.changes_to_commits}, 'changes')
            self.changes_to_commits = []

    def gather_func_to_commits(self, commit, modified_functions):
        for method in modified_functions:
            method_name = method['function name']
            if method_name in self.func_name_to_params_and_commits.keys():
                self.func_name_to_params_and_commits[method_name]['commits that changed in'].append({
                    'commit index': self.commit_index,
                    'commit message': commit.msg})
            else:
                self.func_name_to_params_and_commits[method_name] = {
                    'params': method['function params'],
                    'commits that changed in': [{'commit index': self.commit_index, 'commit message':commit.msg}] }

    def gather_all_functions(self, commit, modified_files):
        add_functions, delete_functions = self.classify_functions(modified_files)
        for method in delete_functions:
            if method in self.existing_functions:
                self.existing_functions.remove(method)
        for method in add_functions:
            self.existing_functions.append(method)

        self.commit_id_to_functions[self.commit_index] = {'hexsha' : commit.hash, 'all functions': self.existing_functions.copy()}

        print(self.commit_index)


    def classify_functions(self,modified_files):
        add_functions = []
        delete_functions = []
        for modified_file in modified_files:
            exist_now = False
            exist_before = False
            methods_before = modified_file.methods_before
            for method in modified_file.changed_methods:
                for old_method in methods_before:
                    if method.name == old_method.name:
                        exist_before = True
                        break
                method_name = self.clear_name(method.name)
                if not exist_before:
                    add_functions.append(method_name)
                    continue

                for new_method in modified_file.methods:
                    if method.name == new_method.name:
                        exist_now = True
                        break
                if not exist_now:
                    delete_functions.append(method_name)
                    continue
        return add_functions, delete_functions

    def extract_modified_functions(self, modified_files):
        list_of_modified_functions = []
        for modified_file in modified_files:
            for method in modified_file.changed_methods:
                list_of_modified_functions.append(
                    {'function name': self.clear_name(method.name),
                     'function params': method.parameters})

        return list_of_modified_functions

    def save_into_file(self, file_name, new_data, dictionary_value):
        if not exists(join(self.data_path, file_name + ".txt")):
            with open(join(self.data_path, file_name + ".txt"), 'w') as outfile:
                json.dump(new_data,outfile, indent=4)

        else:
            with open(join(self.data_path, file_name + ".txt")) as outfile:
                data = json.load(outfile)

            data[dictionary_value].extend(new_data[dictionary_value])

            with open(join(self.data_path, file_name + ".txt"), 'w') as outfile:
                outfile.seek(0)
                json.dump(data,outfile, indent=4)

    def save_functions_per_commit(self, file_name, new_data, dictionary_value):
        data = {dictionary_value: new_data}

        data2 = DataFrame.from_dict(data)
        DataFrame.to_parquet(data2,path=join(self.analysis_path, file_name))

       # with open(join(self.analysis_path, file_name + ".txt"), 'w') as outfile:
       #     json.dump(data, outfile, indent=4)


    def clear_name(self, name):
        if "::" in name:
            return name.split("::")[1]
        else:
            return name


if __name__ == "__main__":
    if len(sys.argv) == 1:
        GatherCommitsData("https://github.com/apache/commons-lang.git","apache_commons-lang-testing-paraquet").gather()
