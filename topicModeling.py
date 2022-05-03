#import pyLDAvis.gensim_models
import gensim
import pickle
from gensim import corpora

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

import nltk
import collections
from spacy.lang.en import English
from sklearn.metrics.pairwise import cosine_similarity
import json
from os.path import exists, join
from os import mkdir
import re
import csv
import warnings
import sys
from pandas import *

# Load the LDA model from sk-learn
warnings.simplefilter("ignore", DeprecationWarning)


parser = English()


nltk.download("wordnet")

nltk.download("stopwords")
nltk.download('omw-1.4')

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append("URL")
        elif token.orth_.startswith("@"):
            lda_tokens.append("SCREEN_NAME")
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(tokens1):
    en_stop = set(nltk.corpus.stopwords.words("english"))
    # tokens = tokenize(text)
    tokens = [token for token in tokens1 if len(token) > 1]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def count_words(words, dict):
    c = collections.Counter(words)
    for w in c:
        if w in dict.keys():
            dict[w]["count files"] += 1
            dict[w]["count total"] += c[w]
        else:
            dict[w] = {"count files": 1, "count total": c[w]}


def remove_number(list):
    new_list = []
    for word in list:
        if word != "":
            try:
                tmp = int(word)
            except:
                new_list.append(word)

    return new_list


def find_average_commit_length(all_commits_for_func, func_to_avg, func_name):
    avg = 0
    for commit in all_commits_for_func:
        seperated = commit.split(sep=" ")
        avg += len(seperated) / len(all_commits_for_func)
    func_to_avg[func_name] = avg


def remove_low_appearence_words(strings, counts):
    words_to_remove = []
    min_apperances = int(len(strings) / 100)+ 1
    for word in counts:
        if int(counts[word]["count total"]) < min_apperances:
            words_to_remove.append(word)
    for dict in strings:
        list = dict["messages"]
        tmp = []
        for word in list:
            if word not in words_to_remove:
                tmp.append(word)
        dict["messages"] = tmp
    return strings


def clean_list_of_strings(unfiltered):
    '''
    this function filter each string '''
    try:
        filtered = list(map(lambda x: re.sub("[./`]", "\n", x), unfiltered))
        filtered = list(map(lambda x: re.sub("[!-,]", "\n", x), filtered))
        filtered = list(map(lambda x: re.sub("[:-@]", "\n", x), filtered))
        filtered = list(map(lambda x: re.sub("[\[-^]", "\n", x), filtered))
        filtered = list(map(lambda x: re.sub("[{-~]", "\n", x), filtered))

        filtered = list(map(lambda x: re.sub("\n+", " ", x), filtered))
        filtered = list(map(lambda x: re.sub("\r+", " ", x), filtered))

        filtered = list(map(lambda x: re.sub(" +", " ", x), filtered))
        filtered = list(map(lambda x: re.sub(" +", " ", x), filtered))

        filtered = list(map(lambda x: x.lower(), filtered))
        filtered = list(map(lambda x: x.split(sep="git-svn")[0], filtered))

        return filtered
    except:
        return []


class TopicModeling:
    def __init__(self, project_name, functions):
        self.project_path = join("projects", project_name)
        self.functions = functions
        self.analysis_path = join(self.project_path, "analysis")

        if not functions:
            self.topics = list(range(20,401,20))
            self.topicModeling_path = join(self.project_path, "topicModelingFiles")
            self.bug_to_sim_name = "bug to file and similarity"
            self.bug_to_sim_path = join(self.topicModeling_path, "bug to file and similarity")
            self.path_to_commit_messages = join(self.analysis_path,"file to commits message.txt")
        else:
            if functions == 1:
                self.topics = list(range(20,201,20))
            elif functions == 2:
                self.topics = list(range(220,401,20))
            else:
                self.topics = list(range(420,601,20))
            self.topicModeling_path = join(self.project_path, "topicModeling")
            self.bug_to_sim_name = "bug to functions and similarity"
            self.bug_to_sim_path = join(self.topicModeling_path, "bug to functions and similarity")
            self.path_to_commit_messages = join(self.analysis_path,"func to commits message.txt")

    def run(self):

        if not (exists(join(self.analysis_path, "func to commits message.txt"))):
            print("missing data")

        else:
            if not self.functions:
                if not exists(join(self.project_path, "Experiments")):
                    mkdir(join(self.project_path, "Experiments"))
                    mkdir(join(self.project_path, "Experiments", "Experiment_1"))
                    mkdir(join(self.project_path, "Experiments", "Experiment_1", "data"))
                    mkdir(join(self.project_path, "Experiments", "Experiment_1", "data", "methods"))


            if not (exists(self.topicModeling_path)):
                mkdir(self.topicModeling_path)


            if not(exists(self.bug_to_sim_path)):
                mkdir(self.bug_to_sim_path)

            if not (exists(join(self.topicModeling_path, "topics"))):
                mkdir(join(self.topicModeling_path, "topics"))

            if not (exists(join(self.topicModeling_path, "filtered_data.txt"))):
                with open(self.path_to_commit_messages) as outfile:
                    data = json.load(outfile)

                with open(join(self.analysis_path, "bugs.txt")) as outfile:
                    all_bugs = json.load(outfile)

                all_bugs = all_bugs['bugs info']['bugs']
                all_bug_to_messages = []

                if self.functions:
                    data = data["functions"]
                else:
                    data = data["files"]
                all_func_to_commit_messages = []
                func_to_avg = {}
                word_to_counts = {}
                # word_to_counts = (
                #     {}
                # )  # word to how many times it showed up and in how many files

                # data cleaning
                for func in data:
                    # returns a list, each element in the list is a filtered commit message
                    filtered = clean_list_of_strings(data[func]["message"])

                    find_average_commit_length(filtered, func_to_avg, func)
                    
                    func_to_commit_messages = {
                        "name": func,
                        "commit_messages": " ".join(str(e) for e in filtered),
                    }

                    # list of dictionaries each represent func and one string of all commit messages
                    all_func_to_commit_messages.append(func_to_commit_messages)

                for bug in all_bugs:
                    # returns a list, each element in the list is a filtered commit message
                    m = bug["description"]
                    if m is None:
                        continue
                    filtered = clean_list_of_strings(m.split())

                    bug_to_messages = {
                        "name": bug,
                        "messages": " ".join(str(e) for e in filtered),
                    }

                    # list of dictionaries each represent func and one string of all commit messages
                    all_bug_to_messages.append(bug_to_messages)

                func_to_prepared_commit_messages = []
               # bug_to_prepared_messages = []

                for func in all_func_to_commit_messages:
                    messages = func["commit_messages"]
                    list_of_words = messages.split(sep=" ")
                    list_of_words_without_numbers = remove_number(list_of_words)

                    # prepared text for lda, removed stop words and small words
                    text_data = prepare_text_for_lda(list_of_words_without_numbers)

                    count_words(text_data, word_to_counts)
                    func_to_prepared_commit_messages.append({"name": func["name"], "messages": text_data})

                func_to_prepared_commit_messages = remove_low_appearence_words(
                    func_to_prepared_commit_messages, word_to_counts
                )

                # for bug in all_bug_to_messages:
                #     messages = bug["messages"]
                #     list_of_words = messages.split(sep=" ")
                #     list_of_words_without_numbers = remove_number(list_of_words)
                #
                #     # prepared text for lda, removed stop words and small words
                #     text_data = prepare_text_for_lda(list_of_words_without_numbers)
                #
                #     bug_to_prepared_messages.append({"name": bug["name"], "messages": text_data})

                self.save_into_file("word_to_counts", word_to_counts, "words")
                self.save_into_file("filtered_data", func_to_prepared_commit_messages, "strings")
                # self.save_into_file("filtered_data_bugs", bug_to_prepared_messages, "strings")
                self.save_into_file("function_to_avg_commit_len", func_to_avg, "function to avg")
                
            else:
                with open(join(self.topicModeling_path, "filtered_data.txt")) as outfile:
                    func_to_prepared_commit_messages = json.load(outfile)[
                        "strings"]

                # with open(join(self.topicModeling_path, "filtered_data_bugs.txt")) as outfile:
                #     bug_to_prepared_messages = json.load(outfile)[
                #         "strings"]

                with open(
                    join(self.topicModeling_path, "word_to_counts.txt")
                ) as outfile:
                    word_to_counts = json.load(outfile)["words"]

            # gather the prepared messages
            prepared_commit_messages = list(dict["messages"] for dict in func_to_prepared_commit_messages)
            # prepared_commit_messages.extend(list(dict["messages"] for dict in bug_to_prepared_messages))

            dictionary = corpora.Dictionary(prepared_commit_messages)
            corpus = [dictionary.doc2bow(text)
                      for text in prepared_commit_messages]
            pickle.dump(
                corpus, open(join(
                    self.topicModeling_path, "corpus.pkl"), "wb")
            )
            dictionary.save(join(
                self.topicModeling_path, "dictionary.gensim"))

            num_topics_to_table = [
                [
                    "num of topics",
                    "bug id",
                    "num of files that changed",
                    "num of files that changed no tests",
                    "first index exist files",
                    "max index exist files",
                    "num of files checked exist files",
                    "all indexes",
                    "first index exist files no tests",
                    "max index exist files no tests",
                    "num of files checked exist files no tests",
                    "all indexes no tests"
                ]
            ]
            for NUM_TOPICS in self.topics:
                ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
                # returns the table of bug to max index for NUM TOPICS
                num_topics_to_table.extend(
                    self.bug_to_func_and_similarity(
                        ldamodel,
                        dictionary,
                        func_to_prepared_commit_messages,
                        NUM_TOPICS,
                    )
                )

                ldamodel.save(
                    join(
                        self.topicModeling_path,
                        "topics",
                        "model" + str(NUM_TOPICS) + ".gensim",
                    )
                )

                # topics = ldamodel.print_topics(num_words=4)
                # for topic in topics:
                #    print(topic)
                print("finished %d topics" % (NUM_TOPICS))

            # after the data of each num_topics is gathered, create the csv table
            if not self.functions:
                self.create_table(num_topics_to_table)

    def bug_to_func_and_similarity(
        self, lda, dictionary, prepared_commit_messages, NUM_TOPICS
    ):
        with open(
            join(self.project_path, "analysis",
                      "bug_to_commit_that_solved.txt")
        ) as outfile:
            bugs = json.load(outfile)["bugs to commit"]

        if not exists(
            join(
                self.topicModeling_path,
                 self.bug_to_sim_name,
                 self.bug_to_sim_name + " " +
                str(NUM_TOPICS) + " topics",
            )
        ):
            all_bugs = []  # del

            bugs_filtered_and_document_topics = []
            for bug in bugs:
                chances = []
                description = clean_list_of_strings([bug["description"]])
                if description != []:
                    description = description[0].split(sep=" ")
                all_bugs.extend(description)

                # topic number to the chance for the bug being in th topic
                topic_to_chances = lda.get_document_topics(
                    bow=dictionary.doc2bow(description), minimum_probability=0
                )
                for tup in topic_to_chances:
                    chances.append(tup[1])
                bugs_filtered_and_document_topics.append(
                    {"bug id": bug["bug id"], "chances": chances}
                )

            func_filtered_and_document_topics = []
            for func in prepared_commit_messages:
                chances = []
                bow = dictionary.doc2bow(func["messages"])
                topic_to_chances = lda.get_document_topics(
                    bow=bow, minimum_probability=0
                )
                for tup in topic_to_chances:
                    chances.append(tup[1])
                func_filtered_and_document_topics.append(
                    {"name": func["name"], "chances": chances}
                )

            i = len(bugs_filtered_and_document_topics)
            bug_to_func_and_similarity = {}

            for bug in bugs_filtered_and_document_topics:
                func_and_similarity = []
                for func in func_filtered_and_document_topics:
                    cos = cosine_similarity(
                        [bug["chances"]], [func["chances"]]
                    ).tolist()[0][0]
                    func_and_similarity.append(
                        (func["name"], str(round(cos, 3))))

                func_and_similarity.sort(key=lambda x: x[1], reverse=True)
                func_and_similarity_with_index = []
                index = 0
                for f_and_s in func_and_similarity:
                    func_and_similarity_with_index.append(
                        [f_and_s[0], f_and_s[1], str(index)]
                    )
                    index += 1
                bug_to_func_and_similarity[
                    bug["bug id"]
                ] = func_and_similarity_with_index
                print(i)
                i -= 1

            print("finished " + str(NUM_TOPICS) + " topics")
            self.save_into_file_sim(
                join(
                     self.bug_to_sim_name,
                     self.bug_to_sim_name + " " +
                    str(NUM_TOPICS)+ " topics" ,
                ),
                bug_to_func_and_similarity,
                "bugs", str(NUM_TOPICS) + " topics"
            )

        # with open(join()(self.project_path , "topicModeling","bug to funcion and similarity","bug to functions and similarity " + str(NUM_TOPICS) + " topics.txt")) as outfile:
        #     bug_to_func_and_similarity = json.load(outfile)['bugs']

        df = read_parquet(
            path=join(
                self.topicModeling_path,
                 self.bug_to_sim_name,
                 self.bug_to_sim_name +" " +
                str(NUM_TOPICS) + " topics",
            )
        )
        bug_to_func_and_similarity = df.to_dict()["bugs"]

        df = read_parquet(
            path=join(self.analysis_path, "commitId to all functions")
        )
        commit_to_exist_functions = df.to_dict()["commit id"]

        # with open(join()(self.project_path ,"analysis","commitId to all functions.txt")) as outfile:
        #     commit_to_exist_functions = json.load(outfile)['commit id']

        return self.fill_table(
            NUM_TOPICS, bugs, bug_to_func_and_similarity, commit_to_exist_functions
        )

    def fill_table(
        self, NUM_TOPICS, bugs, bug_to_func_and_similarity, commit_to_exist_functions
    ):
        ret_list = (
            []
        )  # will hold tuples that each one represent bug id, num of funcs that changed, max index of changed func

        i = 1
        for bug in bugs:
            # if len(bug["function that changed"]) > 10:
            #     continue

            # index_len_all_funcs = self.find_max_index_all_functions( bug, bug_to_func_and_similarity
            # )
            # index_len_all_funcs_no_tests = self.find_max_index_all_functions_no_tests(
            #     bug, bug_to_func_and_similarity
            # )

            exist_files = commit_to_exist_functions[bug["commit number"]]['file to functions']
            exist_files_filtered = {}
            for f in exist_files:
                if exist_files[f] is not None:
                    exist_files_filtered[f] = exist_files[f]
            # for file in exist_files:
            #     exist_files[file] = exist_files[file].tolist()
            index_len_exist_funcs = self.find_max_index_exist_functions(
                bug,
                bug_to_func_and_similarity,
                exist_files_filtered.keys(),
            )
            index_len_exist_funcs_no_tests = (
                self.find_max_index_exist_functions_no_tests(
                    bug,
                    bug_to_func_and_similarity,
                    exist_files_filtered.keys(),
                )
            )

            ret_list.append(
                [
                    NUM_TOPICS,  # how many topics we are using
                    bug["bug id"],  # issue id
                    len(
                        bug["files that changed"]
                    ),  # num of functions that changed in the commit
                    # num of functions that changed in the commit without tests
                    len(
                        list(
                            file
                            for file in bug["files that changed"]
                            if ("test" and "Test") not in file
                        )
                    ),
                    # str(index_len_all_funcs[0]),
                    # str(index_len_all_funcs[1]),
                    # str(index_len_all_funcs_no_tests[0]),
                    # str(index_len_all_funcs_no_tests[1]),
                    str(index_len_exist_funcs[0]),
                    str(index_len_exist_funcs[1]),
                    str(index_len_exist_funcs[2]),
                    str(index_len_exist_funcs[3]),
                    str(index_len_exist_funcs_no_tests[0]),
                    str(index_len_exist_funcs_no_tests[1]),
                    str(index_len_exist_funcs_no_tests[2]),
                    str(index_len_exist_funcs_no_tests[3])
                ]
            )

            print("finished bug number " + str(i))
            i += 1

        return ret_list

    def find_max_index_all_functions(self, bug, bug_to_func_and_similarity):
        max_index = -1
        func_and_similarity_of_bug = bug_to_func_and_similarity[bug["bug id"]].tolist(
        )
        for func in bug["function that changed"]:
            for func_and_similarity in func_and_similarity_of_bug:
                if func["function name"] == func_and_similarity[0]:
                    max_index = max(max_index, int(func_and_similarity[2]))
                    break

        return max_index, len(func_and_similarity_of_bug)

    def find_max_index_all_functions_no_tests(self, bug, bug_to_func_and_similarity):
        max_index_without_test = -1
        func_and_similarity_of_bug = bug_to_func_and_similarity[bug["bug id"]].tolist(
        )

        # filtering all the test functions
        func_and_similarity_of_bug_without_tests = list(
            func
            for func in func_and_similarity_of_bug
            if ("test" and "Test") not in func[0]
        )
        functions_that_changed_no_tests = list(
            func
            for func in bug["function that changed"]
            if ("test" and "Test") not in func["function name"]
        )

        if len(functions_that_changed_no_tests) == 0:
            return -1, len(func_and_similarity_of_bug_without_tests)

        for func in functions_that_changed_no_tests:
            index = 0
            for func_and_similarity_no_test in func_and_similarity_of_bug_without_tests:
                if func["function name"] == func_and_similarity_no_test[0]:
                    max_index_without_test = max(max_index_without_test, index)
                    break
                index += 1

        return max_index_without_test, len(func_and_similarity_of_bug_without_tests)

    def find_max_index_exist_functions(
        self, bug, bug_to_func_and_similarity, exists_functions
    ):
        max_index_smaller_list = -1

        func_and_similarity_of_bug = (
            bug_to_func_and_similarity[bug["bug id"]].tolist().copy()
        )
        for i in range(len(func_and_similarity_of_bug)):
            func_and_similarity_of_bug[i] = func_and_similarity_of_bug[i].tolist(
            )
        # now im finiding the index only on the list of existing functions in the commit
        exist_funcs_with_similarity = []

        for func_exist in exists_functions:
            func_name = func_exist.split('\\')[-1].split('/')[-1]
            for func_and_similarity in func_and_similarity_of_bug:
                func_sim_name = func_and_similarity[0].split('\\')[-1].split('/')[-1]
                if func_name == func_sim_name:
                    exist_funcs_with_similarity.append([func_sim_name,func_and_similarity[1],func_and_similarity[2]])
                    #exist_funcs_with_similarity.append(func_and_similarity)
                    break
            else:
                pass
        exist_funcs_with_similarity.sort(key=lambda x: x[1], reverse=True)

        min_index = len(exist_funcs_with_similarity)
        all_indexes = []
       # for func in bug["function that changed"]:
        for file in bug["files that changed"]:
            file_name = file.split('\\')[-1].split('/')[-1]
            index = 0
            for exist_func_and_similarity in exist_funcs_with_similarity:
                if file_name == exist_func_and_similarity[0]:
                    max_index_smaller_list = max(max_index_smaller_list, index)
                    min_index = min(min_index, index)
                    all_indexes.append(index)
                    break
                index += 1
            else:
                max_index_smaller_list = max(max_index_smaller_list, index)
                min_index = min(min_index, index)
                all_indexes.append(index)

        return min_index,max_index_smaller_list, len(exist_funcs_with_similarity), all_indexes

    def find_max_index_exist_functions_no_tests(
        self, bug, bug_to_func_and_similarity, exists_functions
    ):
        max_index_smaller_list_no_tests = -1

        func_and_similarity_of_bug = (
            bug_to_func_and_similarity[bug["bug id"]].tolist().copy()
        )
        for i in range(len(func_and_similarity_of_bug)):
            func_and_similarity_of_bug[i] = func_and_similarity_of_bug[i].tolist(
            )
        # now im finiding the index only on the list of existing functions in the commit
        exist_funcs_with_similarity = []

        for func_exist in exists_functions:
            func_name = func_exist.split('\\')[-1].split('/')[-1]
            for func_and_similarity in func_and_similarity_of_bug:
                func_sim_name = func_and_similarity[0].split('\\')[-1].split('/')[-1]
                if func_name == func_sim_name:
                    exist_funcs_with_similarity.append([func_sim_name,func_and_similarity[1],func_and_similarity[2]])
                    #func_and_similarity_of_bug.remove(func_and_similarity)
                    break
        exist_funcs_with_similarity.sort(key=lambda x: x[1], reverse=True)

        exist_funcs_with_similarity_without_tests = list(
            func
            for func in exist_funcs_with_similarity
            if ("test" and "Test") not in func[0]
        )
        functions_that_changed_no_tests = list(
            file
            for file in bug["files that changed"]
            if ("test" and "Test") not in file
        )

        if len(functions_that_changed_no_tests) == 0:
            return -1,-1, len(exist_funcs_with_similarity_without_tests),[]

        min_index = len(exist_funcs_with_similarity_without_tests)
        all_indexes = []
        for file in functions_that_changed_no_tests:
            file_name = file.split('\\')[-1].split('/')[-1]
            index = 0
            for exist_func_and_similarity in exist_funcs_with_similarity_without_tests:
                if file_name == exist_func_and_similarity[0]:
                    max_index_smaller_list_no_tests = max(
                        max_index_smaller_list_no_tests, index
                    )
                    min_index = min(min_index, index)
                    all_indexes.append(index)
                    break
                index += 1
            else:
                max_index_smaller_list_no_tests = max(max_index_smaller_list_no_tests, index)
                min_index = min(min_index, index)
                all_indexes.append(index)

        return min_index, max_index_smaller_list_no_tests, \
               len(exist_funcs_with_similarity_without_tests), all_indexes

    def create_table(self, rows):

        with open(join(self.project_path, "Experiments", "Experiment_1", "data", "topicModeling_indexes.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def save_into_file(self, file_name, new_data, dictionary_value):
        data = {}
        data[dictionary_value] = new_data

        with open(
            join(self.topicModeling_path,
                      file_name + ".txt"), "w"
        ) as outfile:
            json.dump(data, outfile, indent=4)

    def save_into_file_sim(self, path, new_data, dictionary_value, file_name ):
        data = {}
        data[dictionary_value] = new_data

        data2 = DataFrame.from_dict(data)
        data2.to_parquet(
            path=join(self.topicModeling_path, path)
        )
        # data2.to_parquet(
        #     path=join(self.project_path, "Experiments", "Experiment_1", "data", "methods", file_name)
        # )






        # with open(join()(self.project_path , "topicModeling", file_name + ".txt"), 'w') as outfile:
        #     json.dump(data, outfile, indent=4)

    # def visualize(self, NUM_TOPICS):
    #
    #     dictionary = gensim.corpora.Dictionary.load(
    #         join(self.project_path, "topicModeling",
    #                   "dictionary.gensim")
    #     )
    #     corpus = pickle.load(
    #         open(join(self.project_path,
    #                        "topicModeling", "corpus.pkl"), "rb")
    #     )
    #     lda = gensim.models.ldamodel.LdaModel.load(
    #         join(
    #             self.project_path,
    #             "topicModeling",
    #             "topics",
    #             "model" + str(NUM_TOPICS) + ".gensim",
    #         )
    #     )
    #     lda_display = pyLDAvis.gensim_models.prepare(
    #         lda, corpus, dictionary, sort_topics=False
    #     )
    #     pyLDAvis.display(lda_display)


if __name__ == "__main__":
    project_name = "Codec"
    functions = 0
    if len(sys.argv) == 3:
        project_name = str(sys.argv[1])
        functions = int(int(sys.argv[2]))
    TopicModeling(project_name,functions).run()
