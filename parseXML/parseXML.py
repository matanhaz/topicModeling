import xml.etree.ElementTree as ET
import os
import sys

def parse_jira():

    tree = ET.parse('SearchRequest.xml')
    root = tree.getroot()
    titles = []
    for i in root[0].findall('item'):
        titles.append(i.find('key').text)

    if not os.path.exists('bugs'):
        os.mkdir('bugs')

    for item in titles:
        tree = ET.parse('SearchRequest.xml')
        root = tree.getroot()
        for i in root[0].findall('item'):
            if i.find('key').text != item:
                root[0].remove(i)


        tree.write(f'bugs\\{item}.xml')

import os
import pprint
import tqdm
import json




def parse_defects4j(path_to_xml):
    func_name_to_id = {}
    test_to_functions = {}
    id = 0
    for file in tqdm.tqdm(os.listdir(path_to_xml)):
        if not file.endswith('.xml'):
            continue
        test_name = file.split('.xml')[0] + '()'
        test_name = test_name.replace('_', '.', 1)
        test_to_functions[test_name] = []
        tree = ET.parse(os.path.join(path_to_xml, file))
        root = tree.getroot()
        packages = root[1]
        #packages = packages[0]
        for package in packages:
            for classs in package[0]:
                #print(classs.attrib['name'])
                for method in classs[0]:
                    if 'init' in method.attrib['name'] or method.attrib["line-rate"] == '0.0':
                        continue
                    sig = method.attrib['signature']
                    sig = sig.split('(')[1]
                    sig = sig.split(')')[0]
                    sig = sig.split(';')
                    final_sig = ''
                    for arg in sig:
                        if len(arg) > 1:
                            final_sig += arg.split('/')[-1]
                            final_sig += ";"
                    final_sig = final_sig[0:-1]
                    full_func_name = f'{classs.attrib["name"]}.{method.attrib["name"]}({final_sig})'
                    func_id = -1
                    if full_func_name in func_name_to_id.keys():
                        func_id = func_name_to_id[full_func_name]
                    else:
                        func_id = id
                        func_name_to_id[full_func_name] = func_id
                        id += 1
                    test_to_functions[test_name].append(func_id)

              #  print(full_func_name)
              #  print(method.attrib["line-rate"])
 #   pprint.pprint(test_to_functions)
 #   pprint.pprint(func_name_to_id)

    return test_to_functions, func_name_to_id


def create_matrixes(base_project_path, project_matrixes_final_path):
    for folder in os.listdir(base_project_path):
        if folder.endswith('.ini'):
            continue
        matrix = {"bugs": [], "components_names": [], "tests_details": [], "initial_tests": []}
        os.mkdir(os.path.join(project_matrixes_final_path, f"tracer_info_{folder}"))
        failed_tests = []
        with open(os.path.join(base_project_path, folder, "failed_tests.txt"), 'r') as f:
            while True:
                failed_test = f.readline()
                if failed_test == '':
                    break
                failed_test = failed_test.replace(' ', '.')
                failed_test = failed_test.replace('\n', '')
                failed_test += '()'
                failed_tests.append(failed_test)


        with open(os.path.join(base_project_path, folder, "ground_truth.txt"), 'r') as f:
            faulty_function = f.read()

        faulty_function = faulty_function.replace('::', '.')
        faulty_function = faulty_function.replace('\n', '')
        test_to_functions, func_name_to_id = parse_defects4j(os.path.join(base_project_path, folder, "traces"))

        possible_faulty_function_name_old = [func_name for func_name in func_name_to_id.keys() if faulty_function in func_name]
        if len(possible_faulty_function_name_old) == 1:
            faulty_function = possible_faulty_function_name_old[0]
        else:
            possible_faulty_function_name = set()
            for failed_test in failed_tests:
                possible_faulty_function_name.update([func_name for func_name in possible_faulty_function_name_old if func_name_to_id[func_name] in test_to_functions[failed_test]])

            faulty_function = possible_faulty_function_name.pop() if len(possible_faulty_function_name) > 0 else possible_faulty_function_name_old[0]

        # else:
        #     remaining_faulty_functions = []
        #     #possible_faulty_function_name_indexes = [func_name_to_id[func] for func in possible_faulty_function_name]
        #     for failed_test in failed_tests:
        #         remaining_faulty_functions.append([func for func in possible_faulty_function_name if func_name_to_id[func] in test_to_functions[failed_test]])
        #
        #     if len(possible_faulty_function_name) == 1:
        #         faulty_function = possible_faulty_function_name[0]
        matrix['bugs'].append(faulty_function)
        for name, id in func_name_to_id.items():
            matrix['components_names'].append([id, name])

        for test, functions in test_to_functions.items():
            matrix['tests_details'].append([test, functions, 1 if test in failed_tests else 0])

        matrix['initial_tests'] = list(test_to_functions.keys())

        final_matrix = str(matrix)
        final_matrix = final_matrix.replace('\'', '\"')
       # json_object = json.dumps(matrix, indent=4)
        with open(os.path.join(project_matrixes_final_path, f"tracer_info_{folder}", f'matrix_{folder}_full.json'), 'w') as f:
            f.write(final_matrix)



if __name__ == '__main__':
    path = r"G:\.shortcut-targets-by-id\13U4vP4YR04YF6tbuAGRWzFykivjXxRon\Data Diagnosis"
    matrixes_final_path = r"C:\Users\matan\Desktop\thesis_new\matrixes_d4j"
    if len(sys.argv) == 2:
        path = str(sys.argv[1])
    for project in os.listdir(path):
        if os.path.exists(os.path.join(matrixes_final_path, project)):
            continue
        project_matrixes_final_path = os.path.join(matrixes_final_path, project)
        os.mkdir(os.path.join(matrixes_final_path, project))
        create_matrixes(os.path.join(path, project), project_matrixes_final_path)
