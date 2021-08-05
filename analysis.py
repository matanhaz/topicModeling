
import javalang
from git import Repo
import os
import json

PROJECT_PATH = os.getcwd() + "\\projects\\"


def main(run_all, project_name, git_url):
    global PROJECT_PATH
    PROJECT_PATH += project_name

    if run_all:
        if not (os.path.exists(PROJECT_PATH + "\\analysis")):
            os.mkdir(PROJECT_PATH + "\\analysis")

        # analyse_failed_commits()
        function_to_all_messages()
        count_bugs()
        bug_to_commit_that_solved()
        # get_all_functions_on_commit_and_id_to_hexsha(git_url)

    else:
        if not (os.path.exists(PROJECT_PATH + "\\analysis")):
            os.mkdir(PROJECT_PATH + "\\analysis")

        val = int(input("1.analyse_failed_commits\n"
                        "2.function_to_all_messages\n"
                        "3.count_bugs\n"
                        # "4.get_all_functions_on_commit_and_id_to_hexsha\n"
                        "5.bug_to_commit_that_solved"))
        if val == 1:
            analyse_failed_commits()

        elif val == 2:
            function_to_all_messages()

        elif val == 3:
            count_bugs()

        # elif val == 4:
            # get_all_functions_on_commit_and_id_to_hexsha()

        elif val == 5:
            bug_to_commit_that_solved()


# for each file saving how many times it failed and in which commit
def analyse_failed_commits():
    if not (os.path.exists(PROJECT_PATH + "\\data\\failed commits.txt")):
        print("missing failed commits data to analyse")
        return

    file_to_fails = {}

    with open(PROJECT_PATH + "\\data\\failed commits.txt") as outfile:
        data = json.load(outfile)

    failed_commits = data['failed']
    for failed in failed_commits:
        path = failed['file_path']
        file_name = get_file_name(path)

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

        save_into_file("file to fails", file_to_fails, 'files')


def function_to_all_messages():
    if not (os.path.exists(PROJECT_PATH + "\\data\\funcToCommits.txt")):
        print("missing funcToCommits data to analyse")
        return

    func_to_commits_message = {}

    with open(PROJECT_PATH + "\\data\\funcToCommits.txt") as outfile:
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

    save_into_file("func to commits message",
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
    save_into_file("functions counters", dict, 'counters')


def count_bugs():
    if not (os.path.exists(PROJECT_PATH + "\\data\\issues.txt")):
        print("missing issues data to analyse")
        return

    with open(PROJECT_PATH + "\\data\\issues.txt") as outfile:
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
    save_into_file("bugs", result, 'bugs info')


def bug_to_commit_that_solved():
    if not (os.path.exists(PROJECT_PATH + "\\analysis\\bugs.txt")):
        print("missing bugs data to analyse")
        return
    if not (os.path.exists(PROJECT_PATH + "\\data\\data0.txt")):
        print("missing commits data to analyse")
        return

    if not (os.path.exists(PROJECT_PATH + "\\analysis\\commitId to all functions.txt")):
        print("missing commits data to analyse")
        return

    with open(PROJECT_PATH + "\\analysis\\bugs.txt") as outfile:
        data = json.load(outfile)

    with open(PROJECT_PATH + "\\analysis\\commitId to all functions.txt") as outfile:
        commitId_to_hexsha = json.load(outfile)['commit id']

    bugs = data['bugs info']['bugs']
    bugs_id_list = []
    for bug in bugs:
        bugs_id_list.append((bug['issue_id'], bug['description']))

    counter = 0
    commits = {}
    bug_to_commit_that_solved = []

    while (os.path.exists(PROJECT_PATH + "\\data\\data{}.txt".format(counter))):
        with open(PROJECT_PATH + "\\data\\data{}.txt".format(counter)) as outfile:
            data = json.load(outfile)
        data = data['changes']
        for commit in data:
            if not commit['commit_id'] in commits.keys():
                commits[commit['commit_id']] = {
                    'commit_summary': commit['commit_summary'],
                    'functions': commit['functions']
                }
            else:
                commits[commit['commit_id']]['functions'].extend(
                    commit['functions'])
        counter += 1

    for commit_id in commits:
        for id in bugs_id_list:
            if id[0] in commits[commit_id]['commit_summary'] and not_followed_by_a_number(id[0], commits[commit_id]['commit_summary']):
                bug_to_commit_that_solved.append({
                    'bug id': id[0],
                    'hexsha': commitId_to_hexsha[str(commit_id)]['hexsha'],
                    'description': id[1],
                    'commit number': commit_id,
                    'function that changed': commits[commit_id]['functions']
                })
                bugs_id_list.remove(id)

                break

    save_into_file("bug_to_commit_that_solved",
                   bug_to_commit_that_solved, 'bugs to commit')
    print(len(bugs_id_list))


# def get_all_functions_on_commit_and_id_to_hexsha(git_url):
#     try:
#         if not (os.path.exists(PROJECT_PATH + "\\Project")):
#             Repo.clone_from(
#                 git_url, PROJECT_PATH + "\\Project")
#         repo = Repo(PROJECT_PATH + "\\Project")
#     except Exception as e:
#         print("bad Git URL")
#         return
#
#     with open(PROJECT_PATH + "\\analysis\\commitId to all functions2.txt") as outfile:
#         data = json.load(outfile)['commit id']
#
#     all_commits = list(repo.iter_commits('master'))  # list of commits
#
#     git = repo.git
#
#     cur_commit = 0
#
#     commitId_to_all_functions = {}
#     commit_id_to_hexsha = {}
#     while cur_commit <= len(all_commits) - 1:
#         print(cur_commit)
#         git.checkout('-f', all_commits[cur_commit].hexsha)
#         # existing_files_in_commit = git.execute(['git','ls-tree','--name-only','-r',all_commits[cur_commit].hexsha ]).split()
#         # all_functions = []
#         # for file in existing_files_in_commit:
#         #     if ".java" in file and os.path.exists(os.getcwd() + "\\Project\\"+file):
#         #         try:
#         #             tmp_tree = javalang.parse.parse(open(os.path.normpath(os.path.join(repo._working_tree_dir, file))).read())
#         #             all_functions.extend(get_methods_from_tree(tmp_tree))
#         #         except Exception as e:
#         #             continue
#         commitId_to_all_functions[str(len(all_commits)-cur_commit)] = {
#             'hexsha': all_commits[cur_commit].hexsha,
#             'all functions': data[str(len(all_commits)-cur_commit)]}
#         # commit_id_to_hexsha[str(len(all_commits)-cur_commit)] = {
#         #     'hexsha': all_commits[cur_commit].hexsha
#         # }
#
#         cur_commit += 1
#         print(cur_commit)
#
#     save_into_file('commitId to all functions',
#                    commitId_to_all_functions, 'commit id')
#     # save_into_file('commitId to hexsha',
#     #                commit_id_to_hexsha, 'commit id')


def get_methods_from_tree(tree):
    types = tree.types
    list_of_methods_names = list()
    for node in types:
        for m in node.methods:
            list_of_methods_names.append(m.name)
    return list_of_methods_names


def get_file_name(file_path):
    file_name = file_path.split(sep="/")
    file_name = file_name[len(file_name) - 1]
    return file_name


def save_into_file(file_name, new_data, dictionary_value):
    data = {}
    data[dictionary_value] = new_data
    with open(PROJECT_PATH + "\\analysis\\" + file_name + ".txt", 'w') as outfile:
        json.dump(data, outfile, indent=4)


def not_followed_by_a_number(id, description):
    for i in range(0, 10):
        if (id + str(i)) in description:
            return False

    return True


if __name__ == "__main__":
    main(False)
