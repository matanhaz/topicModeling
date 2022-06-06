import csv
import os


file_name = 'data_all_methods_combined.csv'
projects_dir_path = 'projects'

path1 = os.path.join('Experiments', 'Experiment_1', 'data', file_name)
path2 = os.path.join('Experiments', 'Experiment_2', 'data', file_name)
path3 = os.path.join('Experiments', 'Experiment_4', 'data', file_name)
path4 = os.path.join('barinel', 'matrixes_details.csv')


def combine(path, file_name):
    first_project = True
    all_rows = []
    for project in os.listdir(projects_dir_path):
        if os.path.isdir(os.path.join(projects_dir_path,project)):
            with open(os.path.join(projects_dir_path, project,path)) as outfile:
                rows = list(csv.reader(outfile))
            if first_project:
                first_project = False
            else:
                rows = rows[1:]

            all_rows.extend(rows)

    with open(os.path.join(projects_dir_path, file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_rows)


combine(path1, "data_all_projects_localization.csv")
combine(path2, "data_all_projects_diagnosis.csv")
combine(path3, "data_all_projects_diagnosis_files_sim.csv")
combine(path4, "data_all_projects_matrixes_details.csv")

