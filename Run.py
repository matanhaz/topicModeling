from os.path import exists, join
from os import mkdir
import sys
import json
from gather_commits_data import GatherCommitsData
from jiraPart import GatherJiraData
from analysis import analyzer
from topicModeling import TopicModeling


def main():
    if not (exists("project_info.txt")):
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

    if not exists("projects"):
        mkdir("projects")
    if not exists(join("projects", selected_project)):
        mkdir(join("projects", selected_project))

    print("**********Gathering commits data**********")
    GatherCommitsData(git_url, selected_project).gather()
    print("**********Gathering Jira data**********")
    GatherJiraData(jira_url, project, selected_project).gather()
    print("**********Running some analysis**********")
    analyzer(selected_project).run()
    print("**********now comes topic modeling**********")
    TopicModeling(selected_project).run()


if __name__ == "__main__":
    main()
