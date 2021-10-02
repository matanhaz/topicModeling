
import json
import os
import csv

import pandas

from sfl_diagnoser import main as D
from datetime import datetime

from sfl_diagnoser.sfl.Diagnoser.diagnoserUtils import write_json_planning_file, read_json_planning_instance, read_json_planning_file
from sfl_diagnoser.sfl.Diagnoser.Experiment_Data import Experiment_Data
from sfl_diagnoser.sfl.Diagnoser.Diagnosis_Results import Diagnosis_Results


HEXSHAS = os.listdir(os.getcwd()+"\\hexshas\\old")

get_func_name = lambda x: x.split('.')[-1].split('(')[0]



def main():
    selection = input("""1. get data by percentage
             2. create new matrixes""")

    if selection == str(1):
        get_data_by_percentage()
    if selection == str(2):
        check_time_difference()


def check_time_difference():
    if not os.path.exists(os.getcwd() + "\\hexshas\\new"):
        print("missing data")
        return
    
    #hexsha_to_num_topics_and_time = {}
    rows = hexsha_to_num_topics_and_time = [
                    ['percent',
                    'hexsha',
                    'num topics',
                    'time old',
                    'time new' ,
                    'time diff',
                    'precision old',
                    'recall old',
                    'precision new',
                    'recall new',
                    'wasted old',
                    'wasted new',
                    'precision diff',
                    'recall diff',
                    'wasted diff' ]]

    for HEXSHA in HEXSHAS:
        with open(os.getcwd() + "\\hexshas\\old\\" + HEXSHA) as outfile:
            old_matrix = json.load(outfile)
        #hexsha_to_num_topics_and_time[str(HEXSHA)] = {}
        for percent in range(10,31,5):
            for topic in range(20,26):
                start_old = datetime.now()
                ei_old = D.main(HEXSHA , 'old')
                end_old = datetime.now()

                results_old = Diagnosis_Results(ei_old.diagnoses, ei_old.initial_tests, ei_old.error, ei_old.pool, ei_old.get_id_bugs()).metrics

                if results_old['precision'] == 0:
                    break

                start_new = datetime.now()
                try:
                    ei =  D.main(HEXSHA + '_' + str(topic) + '_' + str(percent) , 'new')
                except:
                    end_new = datetime.now()
                    results_new= {
                        'precision':0,
                        'recall': 0,
                        'wasted':len(old_matrix['components_names'])
                    }

                else:
                    end_new = datetime.now()
                    results_new = Diagnosis_Results(ei.diagnoses, ei.initial_tests, ei.error, ei.pool, ei.get_id_bugs()).metrics
                    
                hexsha_to_num_topics_and_time.append([
                    str(percent),
                    str(HEXSHA),
                    str(topic),
                    str(end_old-start_old),
                    str(end_new-start_new),
                    str((end_old-start_old) - (end_new-start_new)),
                    results_old['precision'],
                    results_old['recall'] ,
                    results_new['precision'],
                    results_new['recall'] ,
                    results_old['wasted'],
                    results_new['wasted'],
                    str(results_old['precision'] - results_new['precision']),
                    str(results_old['recall'] - results_new['recall']),
                    str(results_old['wasted'] - results_new['wasted'])
                ])

                # hexsha_to_num_topics_and_time[str(HEXSHA)][topic] = {
                #     'time old' : str(end_old-start_old),
                #     'time new' : str(end_new-start_new),
                #     'precision old' : results_old['precision'],
                #     'recall old' : results_old['recall'] ,
                #     'precision new' : results_new['precision'],
                #     'recall new' : results_new['recall'] ,
                #     'wasted' : results_new['wasted']
                # }
                print(topic)


    # with open(os.getcwd() +"\\diagnose\\data.txt", 'w') as outfile:
    #     json.dump(hexsha_to_num_topics_and_time,outfile, indent=4)

    create_table(rows,'data')
   

def get_data_by_percentage():
    commit_number = 5
    num_topics_to_table = [
                    ['matrix name',
                    'num of topics' ,
                    'percent chosen from exist functions',
                    'percent of functions in matrix',
                    'percent of functions in diagnoses',
                    'percent of functions ןn final answer' ]] # how many functions that realy contained bug was selected
    for HEXSHA in HEXSHAS:
        commit_number += 1
        # change names
        with open(os.getcwd() + "\\hexshas\\old\\" + HEXSHA) as outfile:
                data = json.load(outfile)

        for func in data['components_names']:
            func[1] = get_func_name(func[1])

        bugs = []
        for func in data['bugs']:
            bugs.append(get_func_name(func))
        data['bugs'] = bugs


        # get commit id and exist functions
        # with open(os.getcwd() + "\\analysis\\commitId to all functions.txt") as outfile:
        #     commitId_to_all_functions = json.load(outfile)['commit id']

        df = pandas.read_parquet(path=os.getcwd() + "\\analysis\\commitId to all functions")
        commitId_to_all_functions =df.to_dict()['commit id']

        for id in commitId_to_all_functions:
            if commitId_to_all_functions[id]['hexsha'] == HEXSHA:
                commit_id = str(int(id)+1)
                exist_functions = commitId_to_all_functions[id]['all functions'].tolist()
                break


        # get id of the fixed bug
        with open(os.getcwd() + "\\analysis\\bug_to_commit_that_solved.txt") as outfile:
            bug_to_commit = json.load(outfile)['bugs to commit']

        for bug in bug_to_commit:
            if bug['commit number'] == int(commit_id):
                bug_id = bug['bug id']
                break

       
        topics_to_graph = {}


        for num_topics in range(20,26):
            print(num_topics)

            topics_to_graph[str(num_topics)] = {}

            with open(os.getcwd() + "\\topicModeling\\bug to funcion and similarity\\bug to functions and similarity " + str(num_topics) + " topics.txt") as outfile:
                func_and_similarity = json.load(outfile)['bugs'][bug_id]

            exist_functions_with_similarity = exist_functions_and_similarity(func_and_similarity,exist_functions)
            exist_funcs_with_similarity_without_tests = list(func for func in exist_functions_with_similarity if ("test" or "Test") not in func[0])

            num_funcions = len(exist_funcs_with_similarity_without_tests)
            
            for percent in range(10,31,5):
                partial_length = int((num_funcions*percent)/100)
                partial_list = []
                for j in range(0,partial_length):
                    partial_list.append(exist_funcs_with_similarity_without_tests[j])


                partial_list_names = list(x[0].lower() for x in partial_list)

                counter = count_contains(list(x[1].lower() for x in data['components_names']),partial_list_names)
            
                inside = count_contains(list(x.lower() for x in data['bugs']),partial_list_names)

                in_diagnose_percent = percent_in_diagnose(data,partial_list_names,HEXSHA)
                
                
                topics_to_graph[str(num_topics)][str(percent)] = {
                    'func percent': (counter/len(data['components_names'])*100),
                    'answer inside percent' : (inside / len(data['bugs'])) * 100,
                    'in diagnosis' : in_diagnose_percent
                }


                ############################################################
                # this code creates new matrixes based on the selected functions
                if 10 <= percent <= 30 and num_topics >=20:
                    if not (os.path.exists(os.getcwd() + "\\hexshas\\new")):
                        os.mkdir(os.getcwd() + "\\hexshas\\new")

                    new_matrix = data.copy()
                    new_components_names = []

                    for c in new_matrix['components_names']:
                        if c[1] in partial_list_names:
                            new_components_names.append(c)
                    new_matrix['components_names'] = new_components_names

                    exist_func_numbers_in_matrix = list(c[0] for c in new_components_names)

                    for t in new_matrix['tests_details']:
                        new_arr = []
                        for func_number in t[1]:
                            if func_number in exist_func_numbers_in_matrix:
                                new_arr.append(func_number)
                        t[1] = new_arr

                    with open(os.getcwd() +"\\hexshas\\new\\"+HEXSHA + "_" + str(num_topics)+ '_' + str(percent), 'w') as outfile:
                        json.dump(new_matrix,outfile, indent=4)

                    
        for topic in topics_to_graph:
            for percent in topics_to_graph[topic]:
                num_topics_to_table.append([
                    str(HEXSHA),
                    str(topic),
                    str(percent),
                    str(topics_to_graph[topic][percent]['func percent']),
                    str(topics_to_graph[topic][percent]['in diagnosis']),
                    str(topics_to_graph[topic][percent]['answer inside percent'])
                ])

    #create_table(num_topics_to_table, 'table')

    

def count_contains(check_from, big_list):
    count = 0
    for item in check_from:
        if item in big_list:
            count += 1

    return count
        
def percent_in_diagnose(data,partial_list_names,hexsha):
    partial_list_numbers = []

    for name in partial_list_names:
        for func in data['components_names']:
            if name == func[1]:
                partial_list_numbers.append(func[0])
                break
    
    diagnoses = D.main(hexsha,'old').diagnoses
    diagnosis_list_numbers = list(x.diagnosis[0] for x in diagnoses)

    count = count_contains(diagnosis_list_numbers,partial_list_numbers)
    return (count/len(diagnosis_list_numbers) )*100






def exist_functions_and_similarity(all_functions, exists_functions):
    func_and_similarity_of_bug = all_functions.copy()
# now im finiding the index only on the list of existing functions in the commit
    exist_funcs_with_similarity = []

    for func_exist in exists_functions:
        for func_and_similarity in func_and_similarity_of_bug:
            if func_exist == func_and_similarity[0]:
                exist_funcs_with_similarity.append(func_and_similarity)
                func_and_similarity_of_bug.remove(func_and_similarity)
                break
    exist_funcs_with_similarity.sort(key=lambda x: x[1], reverse=True)

    return exist_funcs_with_similarity


def create_table(rows,name):
    if not (os.path.exists(os.getcwd() + "\\diagnose")):
        os.mkdir(os.getcwd() + "\\diagnose")

    with open('diagnose\\'+ name + '.csv', 'w', newline='',encoding='utf-8' ) as file:
        writer = csv.writer(file)
        writer.writerows(rows)




if __name__ == "__main__":
    main()
