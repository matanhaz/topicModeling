import gather_commits_data
import jiraPart
import analysis
import topicModeling
import os
import sys
import json
from gather_commits_data import GatherCommitsData
from jiraPart import GatherJiraData
from analysis import analyzer
from topicModeling import TopicModeling


def main():
    # if not (os.path.exists(os.getcwd() + "\\project info.txt")):
    #     open(os.getcwd() + "\\project info.txt", 'x')
    #
    # with open(os.getcwd() + "\\project info.txt", 'r') as outfile:
    #     try:
    #         data = json.load(outfile)
    #     except:
    #         data = {}
    #
    # while True:
    #     available_projects = list(x for x in data.keys())
    #     selected_project = input("available project are:\n"
    #                             + str(available_projects) +
    #                             "\nselect one, for adding new one write -> new")
    #     if selected_project in available_projects:
    #         break
    #     if selected_project == "new":
    #         while True:
    #             project_name = input("please insert project's name")
    #             if project_name not in available_projects:
    #                 break
    #             print("this project name already exist, please select another one")
    #         git_url = input("please insert project's git url")
    #         jira_url = input("please insert project's JIRA url")
    #         jira_query_symbol = input("please insert project's JIRA query")
    #         data[project_name] = {
    #             'git url': git_url,
    #             'jira url': jira_url,
    #             'jira query symbol': jira_query_symbol
    #         }
    #         with open(os.getcwd() + "\\project info.txt", 'w') as outfile:
    #             json.dump(data, outfile, indent=4)
    if not (os.path.exists("project_info.txt")):
        print("missing project info")
        exit()
    if len(sys.argv) != 2:
        print("missing argument - project name")
        exit()
    selected_project = str(sys.argv[1])
    with open("project_info.txt", 'r') as outfile:
        data = json.load(outfile)

        git_url = data[selected_project]['git url']
        jira_url = data[selected_project]['jira url']
        project = data[selected_project]['project']

        if not os.path.exists("projects"):
            os.mkdir("projects")
        if not os.path.exists(os.path.join("projects" ,selected_project)):
            os.mkdir(os.path.join("projects" ,selected_project))

        print("**********Gathering commits data**********")
        GatherCommitsData(git_url,selected_project).gather()
        print("**********Gathering Jira data**********")
        GatherJiraData(jira_url,project,selected_project).gather()
        print("**********Running some analysis**********")
        analyzer(selected_project).run()
        print("**********now comes topic modeling**********")
        #TopicModeling(selected_project).run()


if __name__ == "__main__":
    main()
