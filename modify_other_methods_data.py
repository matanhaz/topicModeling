import os
import json
import csv
import tqdm


def change_file_presentation():
    PROJECT = "LANG"
    GROUP = "Commons"
    PROJECT_NAME = "BugLocator_Lang"
    dirs = [dir for dir in os.listdir(
        f"projects\\apache_commons-lang\\{PROJECT_NAME}\\{GROUP}\\{PROJECT}") if "output" not in dir]

    bug_files = []
    for dir in dirs:
        path = f"projects\\apache_commons-lang\\{PROJECT_NAME}\\{GROUP}\\{PROJECT}\\{dir}\\recommended"
        bug_files.extend([(f"{PROJECT}-"+bug_file.split('.')[0], path+"\\"+bug_file) for bug_file in os.listdir(
            path)])

    final_dict = {}
    bugs = {}
    for file in bug_files:
        counter = 0
        with open(file[1]) as f:
            bug_name = file[0]
            sim = []
            for line in f.readlines():
                parsed_line = line.split("\t")
                function_name = parsed_line[2].split('.')[-2]
                if 'test' not in function_name and 'Test' not in function_name:
                    sim.append([function_name, float(parsed_line[1]), counter])
                    counter += 1
            bugs[bug_name] = sim


    final_dict['bugs'] = bugs

    with open(f"projects\\apache_commons-lang\\topicModeling\\bug to funcion and similarity\\bug_to_function_and_similarity_{PROJECT_NAME}" + ".txt", 'w') as outfile:
        json.dump(final_dict, outfile, indent=4)

# write final dict to a json file
rows = [
            ['bug id',
             'num of functions that changed no tests' ,
             'max index exist functions no tests',
             'num of functions checked']]

def gather_otherMethod_topK():
    '''first i need to check which bug exist in my project and their project
        then i need to check which functions has benn changed in those bugs
        then i calculate the top k '''

    with open(f"projects\\apache_commons-lang\\analysis\\bug_to_commit_that_solved.txt", 'r') as outfile:
        with open(f"projects\\apache_commons-lang\\analysis\\commitId to all functions.txt", 'r') as f:
            all_bugs = json.load(outfile)['bugs to commit']
            commit_to_exist_functions = json.load(f)['commit id']
            exist_bugs_and_changed_functions = {}
            for bug in all_bugs:
                functions_that_changed = [func['function name'] for func in bug["function that changed"]]
                functions_that_changed_no_tests = [func for func in functions_that_changed if not ('test' in func or 'Test' in func)]
                exists_functions = [commit_to_exist_functions[commit]['all functions'] for commit in commit_to_exist_functions if commit_to_exist_functions[commit]['hexsha'] == bug['hexsha']][0]
                exists_functions_no_tests = [func for func in exists_functions if not ('test' in func or 'Test' in func)]
                exist_bugs_and_changed_functions[bug['bug id']] = {'function that changed':functions_that_changed,
                                                                   'function that changed no tests':functions_that_changed_no_tests,
                                                                   'exists functions': exists_functions,
                                                                   'exists functions no tests': exists_functions_no_tests}
    
    with open(f"projects\\apache_commons-lang\\topicModeling\\bug to funcion and similarity\\bug_to_function_and_similarity_BugLocator_Lang.txt") as outfile:
        all_bugs = json.load(outfile)['bugs']
        exists_bugs = exist_bugs_and_changed_functions.keys()
        for bug in tqdm.tqdm(all_bugs):
            if bug not in exists_bugs:
                continue
            
 #           max_index = -1 if len(exist_bugs_and_changed_functions[bug]['function that changed']) == 0 else max([[function[2] for function in all_bugs[bug] if function[0] == func][0] for func in exist_bugs_and_changed_functions[bug]['function that changed']])
            all_indexes = [[function[2] for function in all_bugs[bug] if function[0] == func] for func in exist_bugs_and_changed_functions[bug]['function that changed no tests']]
            max_index_no_test = -1 if len(all_indexes)==0 else len(exist_bugs_and_changed_functions[bug]['exists functions no tests'])-1 if [] in all_indexes else max(all_indexes)[0]
            rows.append([bug,
                         len(exist_bugs_and_changed_functions[bug]['function that changed no tests']),
                         max_index_no_test,
                         len(exist_bugs_and_changed_functions[bug]['exists functions no tests'])])
            
        with open(f'projects\\apache_commons-lang\\topicModeling\\table_bugLocator.csv', 'w', newline='', ) as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        
         
gather_otherMethod_topK()