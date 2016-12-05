import re
import nltk
from nltk.tag import brill, brill_trainer
from nltk.corpus.reader import PlaintextCorpusReader
import nltk.data
from os import listdir
import itertools
import csv
import collections
from nltk.corpus import gazetteers
from operator import itemgetter
from OrdnanceSurveyNamesAPI import ordnance_survey_location_query

##############################
# PATHS
##############################

path_names = '/Users/c/Documents/Python/nlp/trainData'
path_trainwsjtext1 = '/Users/c/Documents/Python/nlp/trainData/wsj_training/wsj_0001.txt'
path_taggedfolder = '/Users/c/Documents/Python/nlp/trainData/wsj_training/'
allfiles = listdir(path_taggedfolder)

path_testfolder = '/Users/c/Documents/Python/nlp/trainData/wsj_untagged/'

# ENGLISH HONORIFICS
# SOURCE: https://github.com/dariusk/corpora/blob/master/data/humans/englishHonorifics.json
honorifics = '/Users/c/Documents/Python/nlp/honorifics.txt'

##############################
# WRITING TO FILES
##############################

listoftagged_file = open('extractedtags.txt', 'w')
pospatterns_file = open('pospatterns.txt', 'w')

##############################
# DATA STRUCTURES
##############################
def sort_by_freq_desc(seq):
    """
    An efficient way to sort some sequence by frequency
    utilising a Counter object.

    Complexity:
    Calls to counter key - O(1)
    Building the Counter - O(n) one time call

    :param seq: The sequence to be sorted.
    :return: The sorted sequence.
    """
    counts = collections.Counter(seq)
    sorted_seq = sorted(seq, key=counts.get, reverse=True)
    return sorted_seq


def remove_duplicates(seq):
    """
    Removes duplicates from a list, preserving its ordering.
    Runs in O(n) complexity.

    :param seq: The list to be made into a set.
    :return: A list (set) without duplicates.
    """
    seen = list()
    seen_add = seen.append
    return [x for x in seq if not (x in seen or seen_add(x))]

##############################
# FILES READING
##############################


def honorifics_list():
    """
    Extract the honorifics from given file to a list
    :return: The list of honorifics in alphabetical order.
    """
    list_hon = []
    for line in csv.reader(open(honorifics), delimiter=' '):
        if line is not '':
            list_hon += line
    return list_hon

##############################
# NAMED ENTITY RECOGNITION
##############################


def _names_extract():
    """ IN HOUSE METHOD
    Extract names from our corpus to a set
    Uses the three files for male, female, and family names.

    :return: A set of names.
    """
    name_read = PlaintextCorpusReader(path_names, '.*')

    _names_corpus = set()
    for w in (name_read.words('names.male') + name_read.words('names.female') + name_read.words('names.family')):
        _names_corpus.add(w)
    return _names_corpus


def names_extract(listentities):
    names_corpus = set()
    for e in listentities:
        if e[0] == 'PERSON':
            if '}' not in e[1]:
                for n in e[1].split():
                    names_corpus.add(n)
    return names_corpus.union(_names_extract())


def loc_extract(listentities):
    loc_corpus = set()
    for e in listentities:
        if e[0] == 'LOCATION':
            # remove errors i.e only one char locations
            if len(e[1]) > 1:
                loc_corpus.add(e[1])
    return loc_corpus


def org_extract(listentities, extractedlocats):
    org_corpus = set()
    for e in listentities:
        if e[0] == 'ORGANIZATION':
            # remove errors, i.e organisation is location
            if e[1] not in extractedlocats:
                org_corpus.add(e[1])
    return org_corpus


def folder_to_txt_files(folder):
    """ Convert a folder to a list of the files it contains

    :param folder: the folder where the desired files are located

    :return: A list of the files' names
    """
    agg = []
    allfs = listdir(folder)
    for f in allfs:
        if f.endswith('.txt'):
            agg.append(f)
    return agg


def training_extract(foldertag):
    """ Extract training data from files

    Extract all tagged data and outputs it into a specified file.
    Tagged data must be located in a folder and in .txt format

    :param foldertag: the folder where the tagged data files are located

    :return: A text file with all pre-tagged named entities extracted,
    each on a new line.
    """

    enamex_pattern = re.compile(r'<ENAMEX.*?>.*?</ENAMEX>', re.ASCII)
    list_of_files = folder_to_txt_files(foldertag)
    print("Folder files iterated through.\n")

    data = [re.findall(enamex_pattern, nltk.data.load(foldertag + f, format="text")) for f in list_of_files]
    # flatten the resulting in the most efficient manner:
    merged_data = list(itertools.chain.from_iterable(data))
    print("Training data extracted.\n")

    for item in merged_data:
        listoftagged_file.write("%s\n" % item)

    print("Extracted training entities have been written to text file.\n")
    return merged_data

#  TESTING
# print(training_extract(path_taggedfolder))


def tuples_extract(entitylist):
    """
    Convert a list of entities to a list of tuples consisting of the type
    ('category', 'named entity NE', [('part1ofNE', 'NNP'),...], ['POSTag1',...])

    Uses part of speech tagging.

    To access:
    NAMED ENTITY: for loop: tuples_extract(list)[1]

    :param entitylist: The list consisting of pre-tagged entities.
    :return: A list of tuples with a category, a named entity and said POS tags.
    """

    desired_pattern = re.compile(r'[>"].*?[<"]', re.ASCII)

    # Extract to a list of tuples in the format [("TAG",ENTITY)]
    raw_entities = []
    for l in (re.findall(desired_pattern, entity) for entity in entitylist):
        raw_entities.append(list(map(lambda s: (s[:len(s) - 1])[1:], l)))

    # TESTING
    # return raw_entities
    # print(len(raw_entities))

    processed = []
    for e in raw_entities:
        cat = e[0]
        ne = e[1]
        pos_tag = nltk.pos_tag(ne.split())
        tag_and_ne_list = list(map(list, zip(*pos_tag)))
        just_tag_seq = tag_and_ne_list[1]
        processed += [(cat, ne, pos_tag, just_tag_seq)]

    print("Tuples with POS tags extracted.\n")
    return processed

# TESTING
# print(tuples_extract(ex1))
# print(len(tuples_extract(ex1)))


def pos_patterns_by_cat(tuplelist):
    # USED IN GRAMMAR, refer to ASSIGNMENT1.pdf
    # preserve ordering to determine most commonly used
    _person_patterns = []
    _org_patterns = []
    _locat_patterns = []
    for elem in tuplelist:
        cat = elem[0]
        tags = elem[3]
        if cat == 'PERSON':
            _person_patterns += [(cat, tags)]
        if cat == 'ORGANIZATION':
            _org_patterns += [(cat, tags)]
        if cat == 'LOCATION':
            _locat_patterns += [(cat, tags)]

    person_patterns = sorted(_person_patterns, reverse=True)
    org_patterns = sorted(_org_patterns, reverse=True)
    locat_patterns = sorted(_locat_patterns, reverse=True)

    _allpatterns = list(itertools.chain(person_patterns, org_patterns, locat_patterns))
    print("All POS patterns extracted from list of tuples.")

    # Counts second part of tuple
    # Since the list in the tuple is mutable, we take it's tuple
    counts = collections.Counter((x, tuple(y)) for (x, y) in _allpatterns)
    __allpatterns = sorted(counts, key=counts.get, reverse=True)

    # TESTING output w
    # print(__allpatterns)

    # account for noise
    noise = {")", ":", "``"}
    pos_tag = []
    cat = list(map(itemgetter(0), __allpatterns))
    _pos_tag = map(itemgetter(1), __allpatterns)
    for t in _pos_tag:
        if noise.difference(t) == noise:
            item = list(t)
            pos_tag += [item]

    # TESTING - list of strings for categories, list of lists for pos_tags
    # print(firsts)
    # print(pos_tag)

    allpatterns = sorted(list(zip(cat, pos_tag)), key=lambda x: x[0])

    print("Set of POS patterns by category generated.\n")

    # write all patterns to a file
    for pattern in allpatterns:
        pospatterns_file.write("%s\n" % str(pattern))

    return allpatterns


def pos_patterns_by_cat_bylength(tuplelist):
    # USED IN GRAMMAR, refer to ASSIGNMENT1.pdf
    # order by length of the POS tags list.
    _person_patterns = []
    _org_patterns = []
    _locat_patterns = []
    for elem in tuplelist:
        cat = elem[0]
        tags = elem[3]
        if cat == 'PERSON':
            _person_patterns += [(cat, tags)]
        if cat == 'ORGANIZATION':
            _org_patterns += [(cat, tags)]
        if cat == 'LOCATION':
            _locat_patterns += [(cat, tags)]

    person_patterns = sorted(_person_patterns, reverse=True)
    org_patterns = sorted(_org_patterns, reverse=True)
    locat_patterns = sorted(_locat_patterns, reverse=True)

    _allpatterns = list(itertools.chain(person_patterns, org_patterns, locat_patterns))
    print("All POS patterns extracted from list of tuples.")

    # Counts second part of tuple
    # Since the list in the tuple is mutable, we take it's tuple
    counts = collections.Counter((x, tuple(y)) for (x, y) in _allpatterns)
    __allpatterns = sorted(counts, key=counts.get, reverse=True)

    noise = {")", ":", "``"}
    pos_tag = []
    cat = list(map(itemgetter(0), __allpatterns))
    _pos_tag = map(itemgetter(1), __allpatterns)
    for t in _pos_tag:
        if noise.difference(t) == noise:
            item = list(t)
            pos_tag += [item]

    # Sort instead on the length of the list of POS tags, irregardless of categories
    allpatterns = sorted(list(zip(cat, pos_tag)), key=lambda x: len(x[1]), reverse=True)

    return allpatterns


# TESTING
# ex1 = training_extract(path_taggedfolder)
# ex2 = tuples_extract(ex1)
# names = names_extract(ex2)
# locations = loc_extract(ex2)
# print(locations)
# orgs = org_extract(ex2,locations)
# print(orgs)
# print(len(orgs))
# print(len(names))
# print(len(locations))
# print(pos_patterns_by_cat(ex2))
# print(pos_patterns_by_cat_bylength(ex2))


def org_common_endings(listorgs):
    """
    Get a list of the most common organisation endings e.g Inc., Corp. etc.
    in descending order of their frequency in the training data.

    :param listorgs: A list of organisations to be passed.
    :return: Most common suffixes in organisation names.
    """

    # TODO FIX SOME OCCURRENCES OF SORTED, WHICH WOULD BENEFIT FROM A FREQUENCY COUNTER
    _endingscommon = list()
    for e in listorgs:
            tokens = e.split()
            # filter out proper organisation names with 1 word only
            if len(tokens) > 1:
                _endingscommon.append(tokens[-1])
    counts = collections.Counter(_endingscommon)
    endingscommon = sorted(_endingscommon, key=counts.get, reverse=True)
    return remove_duplicates(endingscommon)

# TESTING
# print(org_common_endings(orgs))


def nameCheck(e, extractednames):
    name = e.split()

    if name[0] in honorifics_list():
        return True

    for n in name:
        if not n[0].isupper():
            return False
        if n in extractednames:
            return True


def locationCheck(e, extractedlocs):
    if e[0].isupper():
        if e in extractedlocs:
            return True
    # if ordnance_survey_location_query(e) > 0:
    #     return True
    else:
        return False

# TESTING LOCATIONS
# print("Location Testing")
# print(locationCheck("Lithuania"))
# print(locationCheck("Upper Westside"))
# print(locationCheck("Dniepropetrovsk\n"))

# print(locationCheck("High Wycombe"))
# print(locationCheck("Birmingham"))
# print(locationCheck("Aberystwyth\n"))
# print(locationCheck("Wisconsin"))
# print(locationCheck("Freiburg\n"))


def orgCheck(e, extractedorgs):
    if e in extractedorgs:
        return True
    if e.isupper():
        return True

    org_tok = e.split()
    if org_tok[-1] in org_common_endings(extractedorgs):
        return True
    else:
        return False

# TESTING ORGANISATIONS
# print("Organisation Testing")
# print(orgCheck('Aberystwyth University'))
# print(orgCheck('ABOEGINARLY'))
# print(orgCheck('Zimbabwean Chamber of Commerce\n'))

# TESTING PERSON NAMES
# print("Name Testing")
# print(nameCheck('Peter Stephenson'))
# print(nameCheck('John Snow'))
# print(nameCheck('Dr. Peter Leuven'))
# print(nameCheck('Mr. Jones'))


def regexpgrammar(pospatterns):
    '''
    Generate a grammar from a list of tuples, consisiting of category and POS tags.

    :param pospatterns: The list of tuples, where a single tuple has the following form
                        ('LOCATION', ['NNP', 'NNP', 'IN', 'NNP', 'NNP'])
    :return: A list of grammar rules to be passed on to a Regexp parser
    '''

    # Example
    # NP: { < DT | PP\$ > ? < JJ > * < NN >}  # chunk determiner/possessive, adjectives and noun
    # { < NNP > +}  # chunk sequences of proper nouns

    print("Length of pospatterns list is ", len(pospatterns))
    _grammar = []
    for line in pospatterns:
        if "NNN" in line[1] or "NNPS" in line[1] or "NNP" in line[1]:
            _grammar.append(line[0] + ": {<" + "><".join(line[1]) + ">}")
    print("Length of generated grammar is ", len(_grammar))
    # TODO new line between each element, otherwise parser throws typeerror
    grammar = "\n".join(_grammar)
    return grammar

# TESTING
# pos_patterns = pos_patterns_by_cat(ex2)
# print(regexpgrammar(pos_patterns))
# cp = nltk.RegexpParser(regexpgrammar(pos_patterns))
# print(cp)


def cattagger(e, backtrack1, set_people, set_locat, set_orgs):
    _es = e.split()
    es = ''.join(_es)

    if nameCheck(es,set_people):
        return "PERSON"
    if locationCheck(es,set_locat):
        return "LOCATION"
    if orgCheck(es, set_orgs):
        return "ORGANIZATION"

    if backtrack1 is not None:
        if backtrack1 is "in":
            return "LOCATION"
        if backtrack1 in honorifics_list():
            return "PERSON"
    else:
        return None


def namedentityrecognition(trainingpath, testpath):
    """
    A method which perfoms NER given a training path with and a test path.

    :param trainingpath: A path where TAGGED data is located.
    :param testpath: A path where UNTAGGED data is located.

    :return: The list of tagged entities and some statistics on precision.
    """

    ########################
    # SETUP
    ########################

    # Flow:
    # Grammar(lsit of tuples) <- pospatternsbycategory(tuple list) <- tupleextract(entitylist) <-
    # <- training_extract(path_taggedfolder)

    # names_extract(list) <- tupleextract
    # ditto for loc_- and org_extr

    catset = {"PERSON", "ORGANIZATION", "LOCATION"}
    extracted_train = training_extract(trainingpath)
    long_tuples = tuples_extract(extracted_train)

    ########################
    # THE TWO GRAMMARS
    ########################

    # Ordered by frequency of POS patterns
    pos_cat_tuples = pos_patterns_by_cat(long_tuples)

    # Order by length of POS patterns
    # pos_cat_tuples = pos_patterns_by_cat_bylength(long_tuples)

    ########################
    # MORE SETUP
    ########################

    grammar = regexpgrammar(pos_cat_tuples)
    par = nltk.RegexpParser(grammar)

    names_set = names_extract(long_tuples)
    locat_set = loc_extract(long_tuples)
    organis_set = org_extract(long_tuples, locat_set)

    ########################
    # LOAD TEST DATA
    ########################

    all_test_files = folder_to_txt_files(testpath)
    print("Test directory iterated through.\n")

    test_data = [nltk.data.load(testpath + f, format="text") for f in all_test_files]
    print("Test data extracted.\n")

    ########################
    # COUNTERS FOR STATS
    ########################

    good = 0
    bad = 0

    ########################
    # NER
    ########################

    test_data_tostring = "".join(str(i) for i in test_data)
    _test_data_sent_tok = nltk.sent_tokenize(test_data_tostring)
    # remove newlines "\n"
    # strip punctuation at the end position.
    tagged = []
    unsuccessful = []
    test_data_sent_tok = []
    for s in _test_data_sent_tok:
        if "\n" in s:
            ss = s.split("\n")
            test_data_sent_tok += ss

    print("Test sentences have been tokenised.\n")
    print("Please wait until the process has finished.")
    # print(test_data_sent_tok)
    test = []
    for s in test_data_sent_tok:
        _ss = nltk.word_tokenize(s)
        ss = nltk.pos_tag(_ss)

        # Apply chunk parser to a POS tagged sentence
        prs = par.parse(ss)
        es = [(subt.leaves(), subt.label()) for subt in prs.subtrees() if subt.label() in catset]

        named_es = [(" ".join(x[0]), x[1]) for x in [([k[0] for k in d[0]], d[1]) for d in es]]
        test_data = s
        for ne in named_es:
            # prev_words = s.split(ne[0])
            # (head, sep, tail) - [0] [1] [2] for access of element within
            prev_words = s.partition(ne[0])
            # if prev_words[1] == '':
            if prev_words[2] == '':
                prev_word = None
            else:
                head_of_prev = prev_words[0].split()
                if len(head_of_prev) != 0 :
                    prev_word = head_of_prev[-1]
                else:
                    prev_word = None

            cat = cattagger(ne[0], prev_word, names_set, locat_set, organis_set)
            if cat is not None:
                tagged += [(cat, ne[0])]
                test_data = test_data.replace(ne[0], "<ENAMEX TYPE=\"" + cat + "\">" + ne[0] + "</ENAMEX>")
                test += [test_data]
                # print(test_data)
            else:
                unsuccessful.append([ne])

    ########################
    # ACCURACY MEASUREMENTS
    ########################
    delimiter = "-"*80
    print(delimiter)
    print("Some stats:")
    print(delimiter)

    training_set = set()
    for e in long_tuples:
        training_set.add((e[0], e[1]))
    train_length = len(long_tuples)
    print("The length of the training data is", train_length)
    print("The training set size is", len(training_set))

    for e in tagged:
        if e in training_set:
            good += 1
        else:
            bad += 1

    print("FOUND: ", len(tagged))
    print("In comparsion with training set, of those good are: ", good)
    print("ATTEMPTED: ", len(unsuccessful))

    return tagged


# TEST 1 on old data
# namedentityrecognition(path_taggedfolder, path_testfolder)

# TEST 2 on newly released data
newtagged = '/Users/c/Documents/Python/nlp/testData/wsj_test_tagged/'
newuntagged = '/Users/c/Documents/Python/nlp/testData/wsj_New_test_data/'

namedentityrecognition(newtagged,newuntagged)
