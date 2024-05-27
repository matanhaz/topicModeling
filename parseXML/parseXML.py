import xml.etree.ElementTree as ET
import os
import sys

ET.register_namespace('', "http://maven.apache.org/POM/4.0.0")
ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")

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

PRIMITIVES = {'Z': "boolean", 'V': "void", 'I': "int", 'J': "long", 'C': "char", 'B': "byte", 'D': "double",
                  'S': "short", 'F': "float"}

def get_class_constants(file_name, lines):
    file_lines = []
    with open(file_name, 'r') as file:
        file_lines = file.readlines()

    constants = []
    for line in lines:
        line_hits = int(line.attrib['hits'])
        if line_hits == 0:
            continue
        line_number = int(line.attrib['number']) - 1
        full_line = file_lines[line_number]
        index = 1
        while ';' not in full_line:
            full_line += file_lines[line_number + index]
            index += 1
        full_line = full_line.replace("\n", '')
        constants.append(full_line)
    return constants

def get_sig(method):
    sig = method.attrib['signature']
    sig = sig.split('(')[1]
    sig = sig.split(')')[0]
    sig = sig.split(';')
    final_sig = ''
    for arg in sig:
        is_list = False
        while len(arg) > 0:
            if arg[0] == '[':
                is_list = True
            elif arg[0] in PRIMITIVES.keys():
                final_sig += PRIMITIVES[arg[0]]
                if is_list:
                    final_sig += '[]'
                    is_list = False
                final_sig += ";"
            else:
                final_sig += arg.split('/')[-1]
                if is_list:
                    final_sig += '[]'
                    is_list = False
                final_sig += ";"
                break
            arg = arg[1:]

    final_sig = final_sig[0:-1]
    return final_sig

def get_max_hits_in_method_line(lines):
    max_hits = 0
    for line in lines:
        line_hit = int(line.attrib['hits'])
        max_hits = max(max_hits, line_hit)
    return max_hits

def parse_component_to_id(path_to_xml, failed_tests):
    func_name_to_id = {}
    id = 0
    try:
        for file in failed_tests:
            test_name = file.split('.xml')[0] + '()'
            test_name = test_name.replace('_', '.', 1)
            tree = ET.parse(file)
            root = tree.getroot()
            packages = root[1]
            #packages = packages[0]
            for package in packages:
                if package.attrib["line-rate"] == '0.0':
                    continue
                for classs in package[0]:
                    if classs.attrib["line-rate"] == '0.0':
                        continue
                    constants = []
                    for method in classs[0]:
                        if method.attrib["line-rate"] == '0.0':
                            continue

                        final_sig = get_sig(method)
                        func_name = method.attrib["name"]
                        if func_name == '<clinit>':
                            constants = get_class_constants(os.path.join(path_to_xml, "..", "repo", "src", classs.attrib["filename"]), method[0])
                            continue
                        if func_name == '<init>':
                            func_name = classs.attrib["name"].split('.')[-1]
                        full_func_name_without_sig = f'{classs.attrib["name"]}.{func_name}'
                        full_func_name = f'{classs.attrib["name"]}.{func_name}({final_sig})'
                        func_id = -1

                        appears_in_constants = len([c for c in constants if func_name + '(' in c])
                        max_hits_in_method_line = get_max_hits_in_method_line(method[0])

                        if max_hits_in_method_line <= appears_in_constants:
                            continue

                        for func in func_name_to_id.keys():
                            if full_func_name in func:
                                func_id = func_name_to_id[func]
                                func_name_to_id.pop(func)
                                break
                        # if full_func_name in func_name_to_id.keys():
                        #     func_id = func_name_to_id[full_func_name]
                        else:
                            func_id = id
                            id += 1
                        func_name_to_id[full_func_name] = func_id
    except Exception as e:
        print(e)
    return func_name_to_id



def parse_test_to_components(path_to_xml, func_name_to_id):
    test_to_functions = {}
    try:
        for file in tqdm.tqdm(os.listdir(path_to_xml)):
            if not file.endswith('.xml'):
                continue
            test_name = file.split('.xml')[0].split('[')[0] + '()'
            test_name = test_name.replace('_', '.', 1)
            if test_name in test_to_functions.keys():
                continue
            tree = ET.parse(os.path.join(path_to_xml, file))
            root = tree.getroot()
            packages = root[1]
            #packages = packages[0]
            for package in packages:
                if package.attrib["line-rate"] == '0.0':
                    continue
                for classs in package[0]:
                    if classs.attrib["line-rate"] == '0.0':
                        continue

                    #print(classs.attrib['name'])
                    for method in classs[0]:
                        if method.attrib["line-rate"] == '0.0':
                            continue


                        final_sig = get_sig(method)
                        func_name = method.attrib["name"]
                        if '<' in func_name:
                            func_name = classs.attrib["name"].split('.')[-1]
                        full_func_name = f'{classs.attrib["name"]}.{func_name}({final_sig})'
                        if full_func_name in func_name_to_id.keys():
                            if not test_name in test_to_functions.keys():
                                test_to_functions[test_name] = []
                            func_id = func_name_to_id[full_func_name]
                            test_to_functions[test_name].append(func_id)
    except Exception as e:
        print(e)


              #  print(full_func_name)
              #  print(method.attrib["line-rate"])
 #   pprint.pprint(test_to_functions)
 #   pprint.pprint(func_name_to_id)

    return test_to_functions


def create_matrixes(base_project_path, project_matrixes_final_path):
    if not os.path.exists(project_matrixes_final_path):
        os.mkdir(project_matrixes_final_path)
    existing_matrixes = [file.split('_')[2] for file in os.listdir(project_matrixes_final_path)]
    for folder in os.listdir(base_project_path):
        if folder.endswith('.ini') or folder in existing_matrixes:
            continue
        print(folder)
        try:
            matrix = {"bugs": [], "components_names": [], "tests_details": [], "initial_tests": []}
            failed_tests = []
            failed_tests_files = []
            with open(os.path.join(base_project_path, folder, "failed_tests.txt"), 'r') as f:
                while True:
                    failed_test = f.readline()
                    if failed_test == '':
                        break
                    failed_tests2 = failed_test.split(',')
                    failed_tests_files = [failed_test.replace(' ', '_').replace('\n', '').replace('::', '_') + '.xml' for failed_test in failed_tests2]
                    failed_tests2 = [f.replace(' ', '.').replace('\n', '').replace('::', '.') + '()' for f in failed_tests2]

                    failed_tests.extend(failed_tests2)
                    failed_tests_files.extend(failed_tests_files)


            with open(os.path.join(base_project_path, folder, "ground_truth.txt"), 'r') as f:
                faulty_function = f.read()

            faulty_function = faulty_function.replace('::', '.')
            faulty_function = faulty_function.replace('\n', '')

            func_name_to_id = parse_component_to_id(os.path.join(base_project_path, folder, "traces"),[os.path.join(base_project_path, folder, "traces", failed_test_file) for failed_test_file in failed_tests_files ])
            test_to_functions = parse_test_to_components(os.path.join(base_project_path, folder, "traces"), func_name_to_id)

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
            final_matrix = final_matrix.lower()
           # json_object = json.dumps(matrix, indent=4)

            os.mkdir(os.path.join(project_matrixes_final_path, f"tracer_info_{folder}"))
            with open(os.path.join(project_matrixes_final_path, f"tracer_info_{folder}", f'matrix_{folder}_full.json'), 'w') as f:
                f.write(final_matrix)
        except Exception as e:
            print("folder failed:", folder)


if __name__ == '__main__':
    path = r"G:\.shortcut-targets-by-id\13U4vP4YR04YF6tbuAGRWzFykivjXxRon\Data Diagnosis"
    matrixes_final_path = r"C:\Users\matan\Desktop\thesis_new\new matrixes"
    if len(sys.argv) == 2:
        path = str(sys.argv[1])
    for project in ['cli']:
        project_matrixes_final_path = os.path.join(matrixes_final_path, project)
        for _ in range(4):
            create_matrixes(os.path.join(path, project), project_matrixes_final_path)
        break
