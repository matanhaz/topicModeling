import json
import os

from jira import JIRA

DEFAULT_JIRA_URL = "http://issues.apache.org/jira"
DEFAULT_JIRA_QUERY = 'project = LANG and (status = RESOLVED or status = closed)'
BLOCK_SIZE = 100


def main(jira_url,jira_query_symbol, project_name):

    jira_conn = JIRA(jira_url)

    block_size = BLOCK_SIZE
    block_num = 0
    sql_req = "project = " + jira_query_symbol + " and (status = RESOLVED or status = closed)"
    while True:  # while true
        print(block_num)
        start_idx = block_num * block_size
        if block_num == 0:
            issues = jira_conn.search_issues(sql_req, startAt=start_idx, maxResults=block_size)
        else:
            more_issue = jira_conn.search_issues(sql_req, startAt=start_idx, maxResults=block_size)
            if len(more_issue)>0:
                for x in more_issue:
                    issues.append(x)
            else:
                break
        if len(issues) == 0:
            # Retrieve issues until there are no more to come
            break
        block_num += 1

    issues_dict ={}
    issues_dict['issues']=[]

    for issue in issues:
        new_issue = {
            'issue_id': issue.key, 'project': issue.fields.project.name, 'title': issue.fields.summary,
            'type': issue.fields.issuetype.name,'description': issue.fields.description}

        issues_dict['issues'].append(new_issue)

# this part sometimes missings something so if i dont need all field i wont use it
#     for issue in issues:  # saving each issue and his fields
#         if issue.fields.issuetype is None or issue.fields.priority is None or issue.fields.status is None or issue.fields.resolution is None :
#
#             continue
#         versions = []
#         fix_versions = []
#         components = []
#         for version in issue.fields.versions:
#             versions.append(version.name)
#         for fix in issue.fields.fixVersions:
#             fix_versions.append(fix.name)
#         for comp in issue.fields.components:
#             components.append(comp.name)
#         new_issue = {
#             'issue_id': issue.key, 'project': issue.fields.project.name, 'title': issue.fields.summary,
#             'type': issue.fields.issuetype.name, 'priority': issue.fields.priority.name, 'versions': versions,
#             'components': components, 'labels': issue.fields.labels, 'status': issue.fields.status.name,
#             'resolution': issue.fields.resolution.name, 'fix_versions': fix_versions, 'description': issue.fields.description,
#             'created': issue.fields.created, 'updated': issue.fields.updated,'resolved': issue.fields.resolutiondate
#         }
#         issues_dict['issues'].append(new_issue)


    with open(os.getcwd() + "\\projects\\" + project_name + "\\data\\issues.txt", 'w') as outfile:
        json.dump(issues_dict, outfile, indent=4)


if __name__ == "__main__":
    main(DEFAULT_JIRA_URL,DEFAULT_JIRA_QUERY)
