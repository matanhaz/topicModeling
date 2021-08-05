import gather_commits_data
import jiraPart
import analysis
import topicModeling
import os
import json


def main():
    if not (os.path.exists(os.getcwd() + "\\project info.txt")):
        open(os.getcwd() + "\\project info.txt", 'x')

    with open(os.getcwd() + "\\project info.txt", 'r') as outfile:
        try:
            data = json.load(outfile)
        except:
            data = {}

    while True:
        available_projects = list(x for x in data.keys())
        selected_prject = input("available project are:\n"
                                + str(available_projects) +
                                "\nselect one, for adding new one write -> new")
        if selected_prject in available_projects:
            break
        if selected_prject == "new":
            while True:
                project_name = input("please insert project's name")
                if project_name not in available_projects:
                    break
                print("this project name already exist, please select another one")
            git_url = input("please insert project's git url")
            jira_url = input("please insert project's JIRA url")
            jira_query_symbol = input("please insert project's JIRA query")
            data[project_name] = {
                'git url': git_url,
                'jira url': jira_url,
                'jira query symbol': jira_query_symbol
            }
            with open(os.getcwd() + "\\project info.txt", 'w') as outfile:
                json.dump(data, outfile, indent=4)

    git_url = data[selected_prject]['git url']
    jira_url = data[selected_prject]['jira url']
    jira_query_symbol = data[selected_prject]['jira query symbol']

    if not (os.path.exists(os.getcwd() + "\\projects")):
        os.mkdir(os.getcwd() + "\\projects")
    if not (os.path.exists(os.getcwd() + "\\projects\\" + selected_prject)):
        os.mkdir(os.getcwd() + "\\projects\\" + selected_prject)

    print("**********Gathering commits data**********")
    #gather_commits_data.main(git_url,selected_prject)
    print("**********Gathering Jira data**********")
    #jiraPart.main(jira_url,jira_query_symbol,selected_prject)
    print("**********Running some analysis**********")
    #analysis.main(True,selected_prject, git_url)
    print("**********now comes topic modeling**********")
    topicModeling.main(selected_prject)


if __name__ == "__main__":
    main()
