import os
import subprocess
from os.path import exists, join
from os import mkdir
import json
import sys
from pydriller import repository, git
from git import Repo
from copy import deepcopy
from pandas import *


class GatherCommitsData:
    def __init__(self, git_url, project_name):
        self.project_path = join("projects", project_name)
        self.data_path = join(self.project_path, "data")
        self.analysis_path = join(self.project_path, "analysis")
        self.git_repo_path = join(self.project_path, "project")

        self.git = git_url
        self.changes_to_commits = []
        self.functions = []
        self.files = []

        self.func_name_to_params_and_commits = {}
        self.file_name_to_commits = {}
        self.existing_functions = []
        self.existing_files = {}
        self.commit_id_to_functions = {}

        self.commit_index = 1

        self.create_dirs()


    def clone(self):
            if os.listdir(self.git_repo_path) == []:
                try:
                    subprocess.check_output(['git', 'clone', self.git, 'project'], cwd=self.project_path)
                except Exception as e:
                    print(e)
                    raise


    def get_tags(self):
            result = subprocess.check_output(['git', 'tag'], cwd=self.git_repo_path)
            if result is None:
                return None
            tags = result.decode().split('\n')
            return tags


    def filter_tag(self, tag):
        i = -1
        for index, chr in enumerate(tag):
            if chr.isdigit():
                i = index
                break
        new_tag = ''
        while i < len(tag) and (tag[i].isdigit() or tag[i] in ('.', '_')):
            if tag[i] == '_':
                new_tag += '.'
            else:
                new_tag += tag[i]
            i += 1
        return new_tag

    def gather(self):
        self.clone()
        tags = self.get_tags()
        commits_of_tags = [(tag, git.Git(self.git_repo_path).get_commit_from_tag(tag), self.filter_tag(tag)) for tag in tags if tag != '' and ('RC' or 'Rc' or 'rC' or 'rc') not in tag]
        commits_of_tags.sort(key= lambda x: x[1].committer_date)
        cur_index = 0
        last_index = len(commits_of_tags)
        tag_to_hexsha = {}

        repo = repository.Repository(self.git,  only_modifications_with_file_types=['.java'])
        commits = repo.traverse_commits()

        for commit in commits:
            if not commit.in_main_branch:
                self.commit_index += 1
                continue

            if cur_index < last_index and commit.committer_date > commits_of_tags[cur_index][1].committer_date:
                while cur_index < last_index and commit.committer_date > commits_of_tags[cur_index][1].committer_date:
                    tag_to_hexsha[commits_of_tags[cur_index][0]] = {'hash' :commit.hash, 'filtered name': commits_of_tags[cur_index][2]}
                    cur_index += 1

            # data0, data1 ....
            modified_functions = self.extract_modified_functions(commit.modified_files)
            modified_files = [file.new_path for file in commit.modified_files if file.change_type.name == 'MODIFY' and '.java' in file.filename]
            self.gather_commit_changes(commit, modified_functions, modified_files)

            # func to commits
            self.gather_func_to_commits(commit, modified_functions, modified_files)

            # commit id to all functions
            self.gather_all_functions(commit, commit.modified_files)


            self.commit_to_file_and_functions(commit.modified_files)

            self.commit_index += 1



        file_number = self.get_file_number()

        for func in self.func_name_to_params_and_commits.keys():
            self.functions.append({
                'function name' : func,
                'function params': self.func_name_to_params_and_commits[func]['params'],
                'commits that changed in': self.func_name_to_params_and_commits[func]['commits that changed in']
            })
        for file in self.file_name_to_commits.keys():
            self.files.append({
                'file name' : file,
                'commits that changed in': self.file_name_to_commits[file]['commits that changed in']
            })


        function_to_commits_dic = {}
        function_to_commits_dic['functions'] = self.functions
        with open(join(self.data_path, "funcToCommits.txt"), 'w') as outfile:
            json.dump(function_to_commits_dic, outfile, indent=4)

        files_to_commits_dic = {}
        files_to_commits_dic['files'] = self.files
        with open(join(self.data_path, "fileToCommits.txt"), 'w') as outfile:
            json.dump(files_to_commits_dic, outfile, indent=4)

        self.save_into_file("data"+str(file_number), {'changes':self.changes_to_commits}, 'changes')

        with open(join(self.analysis_path, "tag_to_commit_hexsha.txt"), 'w') as outfile:
                json.dump(tag_to_hexsha,outfile, indent=4)

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
        if not exists(self.git_repo_path):
            mkdir(self.git_repo_path)

    def get_file_number(self):
        return int(self.commit_index / 1000)

    def gather_commit_changes(self,commit, list_of_modified_functions, modified_files):
        file_number = self.get_file_number()

        new_change = {
            'commit_id': self.commit_index,
            'hash': commit.hash ,
            'commit_summary': commit.msg,
            'modified_files': modified_files,
            'functions': list_of_modified_functions
        }
        self.changes_to_commits.append(new_change)

        if self.commit_index % 250 == 0:
            self.save_into_file("data"+str(file_number), {'changes':self.changes_to_commits}, 'changes')
            self.changes_to_commits = []

    def gather_func_to_commits(self, commit, modified_functions, modified_files):
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

        for file in modified_files:
            if file in self.file_name_to_commits.keys():
                self.file_name_to_commits[file]['commits that changed in'].append({
                    'commit index': self.commit_index,
                    'commit message': commit.msg})
            else:
                self.file_name_to_commits[file] = {
                    'commits that changed in': [{'commit index': self.commit_index, 'commit message':commit.msg}] }

    def gather_all_functions(self, commit, modified_files):
        add_functions, delete_functions = self.classify_functions(modified_files)

        for method in add_functions:
            self.existing_functions.append(method)

        for method in delete_functions:
            if method in self.existing_functions:
                self.existing_functions.remove(method)

        self.commit_id_to_functions[self.commit_index] = {'hexsha' : commit.hash, 'all functions': self.existing_functions.copy()}

        print(self.commit_index)


    def classify_functions(self,modified_files):
        add_functions = []
        delete_functions = []
        for modified_file in modified_files:
            functions = self.classify_functions_per_file(modified_file)
            add_functions.extend(functions[0])
            delete_functions.extend(functions[1])

        return add_functions, delete_functions

    def classify_functions_per_file(self, modified_file):
        add_functions = []
        delete_functions = []
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
                exist_now = False
                exist_before = False
                continue

            for new_method in modified_file.methods:
                if method.name == new_method.name:
                    exist_now = True
                    break
            if not exist_now:
                delete_functions.append(method_name)
                exist_now = False
                exist_before = False
                continue
            exist_now = False
            exist_before = False
        return add_functions, delete_functions

    def commit_to_file_and_functions(self, modified_files):
        for modified_file in modified_files:
            if '.java' not in modified_file.filename:
                continue

            if modified_file.change_type.name == 'RENAME':
                self.existing_files[modified_file.new_path] = deepcopy(self.existing_files[modified_file.old_path])
                self.existing_files.pop(modified_file.old_path)

            elif modified_file.change_type.name == 'DELETE' and modified_file.old_path in self.existing_files:
                self.existing_files.pop(modified_file.old_path)

            else:
                if modified_file.change_type.name == 'ADD' or modified_file.new_path not in self.existing_files:
                    self.existing_files[modified_file.new_path] = []

                add_functions, delete_functions = self.classify_functions_per_file(modified_file)
                for method in add_functions:
                    self.existing_files[modified_file.new_path].append(method)

                for method in delete_functions:
                    if method in self.existing_files[modified_file.new_path]:
                        self.existing_files[modified_file.new_path].remove(method)

        self.commit_id_to_functions[self.commit_index]['file to functions'] = deepcopy(self.existing_files)




    def extract_modified_functions(self, modified_files):
        list_of_modified_functions = []
        for modified_file in modified_files:
            for method in modified_file.changed_methods:
                for new_method in modified_file.methods:
                    if method.name == new_method.name:
                        list_of_modified_functions.append(
                        {'function name': self.clear_name(method.name),
                        'function params': method.parameters})
                        break



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
        GatherCommitsData("https://github.com/apache/commons-codec.git","Codec").gather()
