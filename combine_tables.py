import csv
import os


all_rows = []
file_name = 'data_all_methods_combined.csv'
projects_dir_path = 'projects'
projects = os.listdir(projects_dir_path)
first_project = True


for project in projects:
    if os.path.isdir(os.path.join(projects_dir_path,project)):
        path = os.path.join(projects_dir_path, project, 'Experiments', 'Experiment_2', 'data', file_name)
        with open(path) as outfile:
            rows = list(csv.reader(outfile))
        if first_project:
            first_project = False
        else:
            rows = rows[1:]

        all_rows.extend(rows)

with open(os.path.join(projects_dir_path, "data_all_projects.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(all_rows)
