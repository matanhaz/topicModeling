import csv
import os
import json

file_name = 'data_all_methods_combined.csv'
projects_dir_path = 'projects'

path1 = os.path.join('Experiments', 'Experiment_1', 'data', file_name)
path2 = os.path.join('Experiments', 'Experiment_2', 'data', file_name)
path2_1 = os.path.join('Experiments', 'Experiment_2', 'data', 'data_all_methods_combined_positive_precision.csv')
path3 = os.path.join('Experiments', 'Experiment_4', 'data', file_name)
path4 = os.path.join('barinel', 'matrixes_details.csv')
path5 = os.path.join('Experiments','Experiment_2', 'data', 'average_results.csv')
path6 = os.path.join('Experiments','Experiment_2', 'results', 'best_topics.txt')
path7 = os.path.join('Experiments','Experiment_2', 'data', 'data_topic_list_comparison.csv')
path8 = os.path.join('Experiments','Experiment_2', 'data', 'Sanity_combined.csv')


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

def combine_JSON(path, file_name):
    combined = {}
    for project in os.listdir(projects_dir_path):
        if os.path.isdir(os.path.join(projects_dir_path,project)):
            with open(os.path.join(projects_dir_path, project,path)) as outfile:
                data = json.load(outfile)
            combined[project] = data

    with open(os.path.join(projects_dir_path, file_name), 'w', newline='') as file:
        json.dump(combined, file, indent=4)


combine(path1, "data_all_projects_localization.csv")
combine(path2, "data_all_projects_diagnosis.csv")
combine(path2_1, "data_all_projects_diagnosis_positive_precision.csv")
#combine(path3, "data_all_projects_diagnosis_files_sim.csv")
combine(path4, "data_all_projects_matrixes_details.csv")
combine(path5, "data_all_projects_MRR_or_avg_sim.csv")
combine_JSON(path6, 'best_topics_all_projects.txt')
combine(path7, 'data_all_topic_list_comparison.csv')
combine(path8, 'data_all_methods_combined_Sanity.csv')
