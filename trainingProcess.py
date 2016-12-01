import re
import nltk
from nltk.tag import brill, brill_trainer
from nltk.corpus.reader import PlaintextCorpusReader
import nltk.data
from os import listdir
import itertools

##############################
# PATHS
##############################

path_names = '/Users/iichr/Documents/Python/nlp/trainData'
path_trainwsjtext1 = '/Users/iichr/Documents/Python/nlp/trainData/wsj_training/wsj_0001.txt'
path_taggedfolder = '/Users/iichr/Documents/Python/nlp/trainData/wsj_training/'

allfiles = listdir(path_taggedfolder)

##############################
# WRITING TO FILES
##############################

listoftagged_file = open('extractedtags', 'w')


def names_extract():
    """ Extract names from our corpus to a list
    Uses the three files for male, female, and family names.

    :return: A list of names.
    """
    # name_read = nltk.corpus.reader.plaintext.PlaintextCorpusReader(path_names, '.*')
    name_read = PlaintextCorpusReader(path_names, '.*')

    names = []
    for w in (name_read.words('names.male') + name_read.words('names.female') + name_read.words('names.family')):
        names.append(['PERSON', w])


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

    data = [re.findall(enamex_pattern, nltk.data.load(foldertag + f, format="text")) for f in list_of_files]
    # flatten the resulting in the most efficient manner:
    merged_data = list(itertools.chain.from_iterable(data))

    for item in merged_data:
        listoftagged_file.write("%s\n" % item)

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
    return processed

# TESTING
print(tuples_extract(ex1))
print(len(tuples_extract(ex1)))


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
