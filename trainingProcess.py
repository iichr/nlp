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

##############################
# PATHS
##############################

path_names = '/Users/iichr/Documents/Python/nlp/trainData'
path_trainwsjtext1 = '/Users/iichr/Documents/Python/nlp/trainData/wsj_training/wsj_0001.txt'
path_taggedfolder = '/Users/iichr/Documents/Python/nlp/trainData/wsj_training/'

allfiles = listdir(path_taggedfolder)

# ENGLISH HONORIFICS
# SOURCE: https://github.com/dariusk/corpora/blob/master/data/humans/englishHonorifics.json
honorifics = '/Users/iichr/Documents/Python/nlp/honorifics.txt'

#

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
    # name_read = nltk.corpus.reader.plaintext.PlaintextCorpusReader(path_names, '.*')
    name_read = PlaintextCorpusReader(path_names, '.*')

    _names_corpus = set()
    for w in (name_read.words('names.male') + name_read.words('names.female') + name_read.words('names.family')):
        # names.add('PERSON', w)
        _names_corpus.add(w)
    return _names_corpus


def names_extract(listentities):
    # noise = ['}', '(', ')']
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


def org_extract(listentities):
    org_corpus = set()
    for e in listentities:
        if e[0] == 'ORGANIZATION':
            # remove errors, i.e organisation is location
            if e[1] not in locations:
                org_corpus.add(e[1])
    return org_corpus


def folder_to_txt_files(folder):
    """ Convert a folder to a list of the files it contains

    :param folder: the folder where the desired files are located

    :return: A list of the files' names
    """
    agg = []
    allfiles = listdir(folder)
    for f in allfiles:
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

# Test pattern to match all words
# enamex_pattern = re.compile(r'<ENAMEX.*?>.*?</ENAMEX>', re.ASCII)
# print(re.findall(enamex_pattern, nltk.data.load(path_trainwsj, format="text")))

#  TESTING
# print(training_extract(path_taggedfolder))
ex1 = training_extract(path_taggedfolder)


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
    # Possible usage in a grammar.
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

    pos_tag = []
    cat = list(map(itemgetter(0), __allpatterns))
    _pos_tag = map(itemgetter(1), __allpatterns)
    for t in _pos_tag:
        item = list(t)
        pos_tag += [item]

    # TESTING - list of strings for categories, list of lists for pos_tags
    # print(firsts)
    # print(pos_tag)

    allpatterns = sorted(list(zip(cat, pos_tag)), key=lambda x: x[0])

    print("Set of POS patterns by category generated.\n")

    # extract all patterns to a file
    for pattern in allpatterns:
        pospatterns_file.write("%s\n" % str(pattern))

    return allpatterns

# TESTING 2 DEC
ex2 = tuples_extract(ex1)
# names = names_extract(ex2)
locations = loc_extract(ex2)
orgs = org_extract(ex2)

# print(len(orgs))
# print(len(names))
# print(len(locations))
# print(pos_patterns_by_cat(ex2))
pos_patterns_by_cat(ex2)


def most_common_endings_org(listorgs):
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
        _endingscommon.append(tokens[-1])
    counts = collections.Counter(_endingscommon)
    endingscommon = sorted(_endingscommon, key=counts.get, reverse=True)
    return remove_duplicates(endingscommon)

# print(most_common_endings_org(orgs))

def regexp_grammar(patternlist):
    return None


def nameCheck(e):
    name = e.split()

    if name[0] in honorifics_list():
        return True

    for n in name:
        if n in names:
            return True
        if not n.isupper():
            return False


def locationCheck(e):
    if e[0].isupper() and e in locations:
        return True


def orgCheck(e):
    if e in orgs:
        return True
    if e.isupper():
        return True

    org_tok = e.split()


# ####################
# Possibly useless   #
# ####################

"""
Sample names tagger
"""

from nltk.corpus import names
from nltk.tag import SequentialBackoffTagger


class NamesTagger(SequentialBackoffTagger):
    def __init__(self, *args, **kwargs):
        SequentialBackoffTagger.__init__(self, *args, **kwargs)
        self.name_set = set([n.lower() for n in names.words()])


def choose_tag(self, tokens, index, history):
    word = tokens[index]
    if word.lower() in self.name_set:
        return 'NNP'
    else:
        return None


"""
Brill tagger wrapper from canvas
"""
# make sure you've got some train_sents!


def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
        brill.Template(brill.Pos([-1])),
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
    ]

    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
    return trainer.train(train_sents, **kwargs)
