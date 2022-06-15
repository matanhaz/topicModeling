from datetime import datetime
import csv
import javalang
from git import Repo
from os.path import exists, join
from os import mkdir
import json
import pandas
from zipfile import ZipFile
#PROJECT_PATH = os.getcwd() + "\\projects\\"



class analyzer:
    def __init__(self, project_name):
        self.project_path = join("projects", project_name)
        self.data_path = join(self.project_path, "data")
        self.analysis_path = join(self.project_path, "analysis")
        self.matrices_path = join(self.project_path, "barinel")
        if not (exists(self.analysis_path)):
            mkdir(self.analysis_path)
        if not (exists(join(self.data_path,'active-bugs.csv'))):
            with ZipFile(join('matrixes',f'{project_name}.zip'), 'r') as zipObj:
                zipObj.extract('active-bugs.csv', path=self.data_path)



    def run(self):
        self.function_to_all_messages()
        self.file_to_all_messages()
        self.count_bugs()
        self.bug_to_commit_that_solved()

    def analyse_failed_commits(self):
        if not (exists(join(self.data_path,"failed commits.txt"))):
            print("missing failed commits data to analyse")
            return

        file_to_fails = {}

        with open(join(self.data_path,"failed commits.txt")) as outfile:
            data = json.load(outfile)

        failed_commits = data['failed']
        for failed in failed_commits:
            path = failed['file_path']
            file_name = self.get_file_name(path)

            exist = file_to_fails.get(file_name)
            if exist == None:
                file_to_fails[file_name] = {
                    'num of fails': 1,
                    'commits info': [{
                        'commit id': failed['commit_id'],
                        'cause of failure': failed['cause of failure']
                    }]
                }
            else:
                file_to_fails[file_name]['num of fails'] += 1
                file_to_fails[file_name]['commits info'].append({
                    'commit id': failed['commit_id'],
                    'cause of failure': failed['cause of failure']
                })

            self.save_into_file("file to fails", file_to_fails, 'files')

    def file_to_all_messages(self):
        if not (exists(join(self.data_path,"funcToCommits.txt"))):
            print("missing funcToCommits data to analyse")
            return

        file_to_commits_message = {}

        with open(join(self.data_path,"fileToCommits.txt")) as outfile:
            data = json.load(outfile)

        files = data['files']

        for file in files:

            name = file['file name']

            exist = file_to_commits_message.get(name)
            if exist == None:
                file_to_commits_message[name] = {'message': []}
            for commit in file["commits that changed in"]:
                file_to_commits_message[name]['message'].append(
                    commit["commit message"])

        self.save_into_file("file to commits message", file_to_commits_message, 'files')


    def function_to_all_messages(self):
        if not (exists(join(self.data_path,"funcToCommits.txt"))):
            print("missing funcToCommits data to analyse")
            return

        func_to_commits_message = {}

        with open(join(self.data_path,"funcToCommits.txt")) as outfile:
            data = json.load(outfile)

        functions = data['functions']

        for func in functions:

            name = func['function name']

            exist = func_to_commits_message.get(name)
            if exist == None:
                func_to_commits_message[name] = {'message': []}
            for commit in func["commits that changed in"]:
                func_to_commits_message[name]['message'].append(
                    commit["commit message"])

        self.save_into_file("func to commits message",
                       func_to_commits_message, 'functions')

        # count how many functions there are, and how many without test functions

        counter = 0
        counter_no_tests = 0

        for func in func_to_commits_message.keys():
            counter += 1
            if 'test' not in func and 'Test' not in func:
                counter_no_tests += 1

        dict = {
            'num of functions': counter,
            'num of functions no tests': counter_no_tests
        }
        self.save_into_file("functions counters", dict, 'counters')


    def count_bugs(self):
        if not (exists(join(self.data_path, "issues.txt"))):
            print("missing issues data to analyse")
            return

        with open(join(self.data_path, "issues.txt")) as outfile:
            data = json.load(outfile)

        counter = 0
        issues = data['issues']
        bugs = []
        for issue in issues:
            if issue['type'] == "Bug":
                counter += 1
                bugs.append(issue)

        result = {}
        result['bug counter'] = counter
        result['bugs'] = bugs
        self.save_into_file("bugs", result, 'bugs info')


    def bug_to_commit_that_solved(self):
        if not (exists( join(self.analysis_path,"bugs.txt"))):
            print("missing bugs data to analyse")
            return
        if not (exists(join(self.data_path,"data0.txt"))):
            print("missing commits data to analyse")
            return

        if not (exists(join(self.analysis_path,"commitId to all functions" ))):
            print("missing commits data to analyse")
            return

        with open(join(self.analysis_path, "bugs.txt")) as outfile:
            data = json.load(outfile)

        with open(join(self.analysis_path, "tag_to_commit_hexsha.txt")) as outfile:
            versions = json.load(outfile)

        with open(join(self.data_path,"active-bugs.csv")) as csvfile:
            bugs = list(csv.reader(csvfile, delimiter=' '))
            active_bugs = []
            for row in bugs:
                active_bugs.append(row[0].split(','))
            avtive_bugs_labels, active_bugs_values = active_bugs[0], active_bugs[1:]

       # with open(join()(self.analysis_path,"commitId to all functions.txt" )) as outfile:
        #    commitId_to_hexsha = json.load(outfile)['commit id']

        df = pandas.read_parquet(path=join(self.analysis_path,"commitId to all functions"))
        commitId_to_hexsha =df.to_dict()['commit id']

        bugs = data['bugs info']['bugs']
        bugs_id_list = []

        for row in active_bugs_values:

            for bug in bugs:
                if row[3].split('-')[1] == bug['issue_id'].split('-')[1]:
                    bugs_id_list.append((bug['issue_id'], bug['description'], bug['versions'], datetime.strptime(bug['resolved'], '%Y-%m-%d'), bug['fixVersions'], row[0]))

        counter = 0
        commits = {}
        bug_to_commit_that_solved = []
        #join(self.analysis_path,f"data{counter}.txt")
        while (exists(join(self.data_path,f"data{counter}.txt"))):
            with open(join(self.data_path,f"data{counter}.txt")) as outfile:
                data = json.load(outfile)
            data = data['changes']
            for commit in data:
                if not commit['hash'] in commits.keys():
                    commits[commit['hash']] = {
                        'id':commit['commit_id'],
                        'commit_summary': commit['commit_summary'],
                        'files': commit['modified_files'],
                        'functions': commit['functions'],
                        'date': commit['date']
                    }
                else:
                    commits[commit['hash']]['functions'].extend(
                        commit['functions'])
                    commits[commit['hash']]['files'].extend(
                        commit['modified_files'])
            counter += 1



        commits_reversed = list(commits.keys())
        #commits_reversed.reverse()
        bug_id_to_changed_functions = {}
        bug_id_to_changed_files = {}
        bug_id_to_fix_hash = {}

        bugs_id_list_copy = bugs_id_list.copy()


        for commit_id in commits_reversed:
            if bugs_id_list_copy == []:
                break

            for row in active_bugs_values:
                if commit_id == row[2]:

                    desc = commits[commit_id]['commit_summary']
                    bug_id_to_changed_functions[row[0]] = commits[commit_id]['functions']
                    bug_id_to_changed_files[row[0]] = commits[commit_id]['files']
                    bug_id_to_fix_hash[row[0]] = commit_id
                  #  bugs_id_list_copy.remove(id)
                  #  break



        # for commit_id in commits_reversed:
        #     if bugs_id_list_copy == []:
        #         break
        #     for id in bugs_id_list_copy:
        #         if id[0] in commits[commit_id]['commit_summary'] and self.not_followed_by_a_number(id[0], commits[commit_id]['commit_summary'])\
        #                 and datetime.strptime(commits[commit_id]['date'], '%Y-%m-%d')>=id[3] and id [0] not in bug_id_to_fix_hash:
        #             desc = commits[commit_id]['commit_summary']
        #             bug_id_to_changed_functions[id[0]] = commits[commit_id]['functions']
        #             bug_id_to_changed_files[id[0]] = commits[commit_id]['files']
        #             bug_id_to_fix_hash[id[0]] = commit_id
        #           #  bugs_id_list_copy.remove(id)
        #           #  break

        for id in bugs_id_list:
            if id[4] != []:
                version = max(id[4])
            else:
                version = max(id[2])
            version = self.filter_tag(version)

            for v in versions:
                if versions[v]['filtered name'] == version and id[5] in bug_id_to_changed_functions.keys():
                    commit_hash = versions[v]['hash']
                    bug_to_commit_that_solved.append({
                        'bug index':id[5],
                        'bug id': id[0],
                        'hexsha': commit_hash,
                        'fix_hash': bug_id_to_fix_hash[id[5]],
                        'description': id[1],
                        'commit number version hash': commits[commit_hash]['id'],
                        'commit number': commits[bug_id_to_fix_hash[id[5]]]['id'],
                        'function that changed': bug_id_to_changed_functions[id[5]],
                        'files that changed': bug_id_to_changed_files[id[5]]
                    })
                    #bugs_id_list.remove(id)

                    break






        self.save_into_file("bug_to_commit_that_solved",
                       bug_to_commit_that_solved, 'bugs to commit')
        print(len(bugs_id_list))
    # def bug_to_commit_that_solved(self):
    #     if not (exists( join(self.analysis_path,"bugs.txt"))):
    #         print("missing bugs data to analyse")
    #         return
    #     if not (exists(join(self.data_path,"data0.txt"))):
    #         print("missing commits data to analyse")
    #         return
    #
    #     if not (exists(join(self.analysis_path,"commitId to all functions" ))):
    #         print("missing commits data to analyse")
    #         return
    #
    #     with open(join(self.analysis_path, "bugs.txt")) as outfile:
    #         data = json.load(outfile)
    #
    #     with open(join(self.analysis_path, "tag_to_commit_hexsha.txt")) as outfile:
    #         versions = json.load(outfile)
    #
    #    # with open(join()(self.analysis_path,"commitId to all functions.txt" )) as outfile:
    #     #    commitId_to_hexsha = json.load(outfile)['commit id']
    #
    #     df = pandas.read_parquet(path=join(self.analysis_path,"commitId to all functions"))
    #     commitId_to_hexsha =df.to_dict()['commit id']
    #
    #     bugs = data['bugs info']['bugs']
    #     bugs_id_list = []
    #     for bug in bugs:
    #         bugs_id_list.append((bug['issue_id'], bug['description'], bug['versions'], datetime.strptime(bug['resolved'], '%Y-%m-%d'), bug['fixVersions']))
    #
    #     counter = 0
    #     commits = {}
    #     bug_to_commit_that_solved = []
    #     #join(self.analysis_path,f"data{counter}.txt")
    #     while (exists(join(self.data_path,f"data{counter}.txt"))):
    #         with open(join(self.data_path,f"data{counter}.txt")) as outfile:
    #             data = json.load(outfile)
    #         data = data['changes']
    #         for commit in data:
    #             if not commit['hash'] in commits.keys():
    #                 commits[commit['hash']] = {
    #                     'id':commit['commit_id'],
    #                     'commit_summary': commit['commit_summary'],
    #                     'files': commit['modified_files'],
    #                     'functions': commit['functions'],
    #                     'date': commit['date']
    #                 }
    #             else:
    #                 commits[commit['hash']]['functions'].extend(
    #                     commit['functions'])
    #                 commits[commit['hash']]['files'].extend(
    #                     commit['modified_files'])
    #         counter += 1
    #
    #
    #
    #     commits_reversed = list(commits.keys())
    #     #commits_reversed.reverse()
    #     bug_id_to_changed_functions = {}
    #     bug_id_to_changed_files = {}
    #     bug_id_to_fix_hash = {}
    #
    #     bugs_id_list_copy = bugs_id_list.copy()
    #     for commit_id in commits_reversed:
    #         if bugs_id_list_copy == []:
    #             break
    #         for id in bugs_id_list_copy:
    #             if id[0] in commits[commit_id]['commit_summary'] and self.not_followed_by_a_number(id[0], commits[commit_id]['commit_summary'])\
    #                     and datetime.strptime(commits[commit_id]['date'], '%Y-%m-%d')>=id[3] and id [0] not in bug_id_to_fix_hash:
    #                 desc = commits[commit_id]['commit_summary']
    #                 bug_id_to_changed_functions[id[0]] = commits[commit_id]['functions']
    #                 bug_id_to_changed_files[id[0]] = commits[commit_id]['files']
    #                 bug_id_to_fix_hash[id[0]] = commit_id
    #               #  bugs_id_list_copy.remove(id)
    #               #  break
    #
    #     for id in bugs_id_list:
    #         if id[4] != []:
    #             version = max(id[4][-1])
    #         else:
    #             version = max(id[2][-1])
    #         version = self.filter_tag(version)
    #
    #         for v in versions:
    #             if versions[v]['filtered name'] == version and id[0] in bug_id_to_changed_functions.keys():
    #                 commit_hash = versions[v]['hash']
    #                 bug_to_commit_that_solved.append({
    #                     'bug id': id[0],
    #                     'hexsha': commit_hash,
    #                     'fix_hash': bug_id_to_fix_hash[id[0]],
    #                     'description': id[1],
    #                     'commit number version hash': commits[commit_hash]['id'],
    #                     'commit number': commits[bug_id_to_fix_hash[id[0]]]['id'],
    #                     'function that changed': bug_id_to_changed_functions[id[0]],
    #                     'files that changed': bug_id_to_changed_files[id[0]]
    #                 })
    #                 #bugs_id_list.remove(id)
    #
    #                 break
    #
    #
    #
    #
    #
    #
    #     self.save_into_file("bug_to_commit_that_solved",
    #                    bug_to_commit_that_solved, 'bugs to commit')
    #     print(len(bugs_id_list))




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

    def get_methods_from_tree(self, tree):
        types = tree.types
        list_of_methods_names = list()
        for node in types:
            for m in node.methods:
                list_of_methods_names.append(m.name)
        return list_of_methods_names


    def get_file_name(self, file_path):
        file_name = file_path.split(sep="/")
        file_name = file_name[len(file_name) - 1]
        return file_name


    def save_into_file(self, file_name, new_data, dictionary_value):
        data = {}
        data[dictionary_value] = new_data

        with open(join(self.analysis_path,f"{file_name}.txt"), 'w') as outfile:
            json.dump(data, outfile, indent=4)


    def not_followed_by_a_number(self, id, description):
        for i in range(0, 10):
            if (id + str(i)) in description:
                return False

        return True





if __name__ == "__main__":
    analyzer("Codec").run()
