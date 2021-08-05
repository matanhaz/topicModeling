import os
import json

from pydriller import repository


#g = git.Git("https://github.com/apache/commons-lang.git")


DEFAULT_GIT_URL = "https://github.com/apache/commons-math.git"
PROJECT_PATH = os.getcwd() + "\\projects\\"


def main(git_url, project_name):
    global PROJECT_PATH
    PROJECT_PATH += project_name

    if not (os.path.exists(PROJECT_PATH)):
        os.mkdir(PROJECT_PATH)

    if not (os.path.exists(PROJECT_PATH + "\\analysis")):
        os.mkdir(PROJECT_PATH + "\\analysis")

    if not (os.path.exists(PROJECT_PATH + "\\data")):
        os.mkdir(PROJECT_PATH + "\\data")

    repo = repository.Repository(git_url)
    commits = repo.traverse_commits()

    file_number = 0
    commit_index = 1

    changes_to_commits = []  # goes to data0, 1 ...
    functions = []  # goes to func to commit

    func_name_to_params_and_commits = {}

    existing_functions = []
    commit_id_to_functions = {}  # commit id to all functions

    for commit in commits:
        if not commit.in_main_branch:
            commit_index += 1
            continue

        modified_files = commit.modified_files
        list_of_modified_functions = extract_modified_files(modified_files)

        changes_to_commits = gather_changes(commit, commit_index, modified_files, changes_to_commits, list_of_modified_functions)

        for method in list_of_modified_functions:
            method_name = method['function name']
            if method_name in func_name_to_params_and_commits.keys():
                func_name_to_params_and_commits[method_name]['commits that changed in'].append({
                    'commit index': commit_index,
                    'commit message':commit.msg})
            else:
                func_name_to_params_and_commits[method_name] = {
                    'params': method['function params'],
                    'commits that changed in': [{'commit index': commit_index, 'commit message':commit.msg}] }

        add_functions, delete_functions = classify_functions(modified_files)
        for method in delete_functions:
            if method in existing_functions:
                existing_functions.remove(method)
        for method in add_functions:
            existing_functions.append(method)

        commit_id_to_functions[commit_index] = {'hexsha' : commit.hash, 'all functions': existing_functions.copy()}


        print(commit_index)
        commit_index += 1

    file_number = (int)(commit_index / 1000)
    
    for func in func_name_to_params_and_commits.keys():
        functions.append({
            'function name' : func,
            'function params': func_name_to_params_and_commits[func]['params'],
            'commits that changed in': func_name_to_params_and_commits[func]['commits that changed in']
        })


    function_to_commits_dic = {}
    function_to_commits_dic['functions'] = functions
    with open(PROJECT_PATH +"\\data\\funcToCommits.txt", 'w') as outfile:
        json.dump(function_to_commits_dic, outfile, indent=4)

    save_into_file("data"+str(file_number), {'changes':changes_to_commits}, 'changes')

    # takes time, maybe seperate to 2 files
    save_functions_per_commit('commitId to all functions',commit_id_to_functions,'commit id')


def classify_functions(modified_files):
    add_functions = []
    delete_functions = []
    for modified_file in modified_files:
        exist_now = False
        exist_before = False
        for method in modified_file.changed_methods:
            for old_method in modified_file.methods_before:
                if method.name == old_method.name:
                    exist_before = True
                    break

            if exist_before == False:
                add_functions.append(clear_name(method.name))
                continue

            for new_method in modified_file.methods:
                if method.name == new_method.name:
                    exist_now = True
                    break
            if exist_now == False:
                delete_functions.append(clear_name(method.name))
                continue
    return add_functions, delete_functions


def extract_modified_files(modified_files):
    list_of_modified_functions = []
    for modified_file in modified_files:
        list_of_modified_functions.extend(list({'function name': clear_name(method.name),
                                                'function params': method.parameters}
                                               for method in modified_file.changed_methods))
    return list_of_modified_functions


def gather_changes(commit, commit_index, modified_files, changes_to_commits, list_of_modified_functions):
    file_number = (int)(commit_index / 1000)


    new_change = {
        'commit_id': commit_index,
        'commit_summary': commit.msg,
        'functions': list_of_modified_functions
    }
    changes_to_commits.append(new_change)

    if commit_index % 250 == 0:
        save_into_file("data"+str(file_number), {'changes':changes_to_commits}, 'changes')
        changes_to_commits = []

    return changes_to_commits


def save_into_file(file_name, new_data, dictionary_value):
    if not (os.path.exists(PROJECT_PATH + "\\data\\"+file_name + ".txt")):
        with open(PROJECT_PATH +"\\data\\"+file_name + ".txt", 'w') as outfile:
            json.dump(new_data,outfile, indent=4)

    else:
        with open(PROJECT_PATH +"\\data\\"+file_name + ".txt") as outfile:
            data = json.load(outfile)

        data[dictionary_value].extend(new_data[dictionary_value])

        with open(PROJECT_PATH +"\\data\\"+file_name + ".txt", 'w') as outfile:
            outfile.seek(0)
            json.dump(data,outfile, indent=4)

def save_functions_per_commit(file_name, new_data, dictionary_value):
    data = {}
    data[dictionary_value] = new_data
    with open(PROJECT_PATH + "\\analysis\\" + file_name + ".txt", 'w') as outfile:
        json.dump(data, outfile, indent=4)


def clear_name(name):
    if "::" in name:
        return name.split("::")[1]
    else:
        print(name)
        return name

if __name__ == "__main__":
    main(DEFAULT_GIT_URL, 'test-math')
