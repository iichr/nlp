import re
import nltk
from nltk.tag import brill, brill_trainer
from nltk.corpus.reader import PlaintextCorpusReader
import nltk.data
from os import listdir
from os import path


path_names = '/Users/iichr/Documents/Python/nlp/trainData'
path_trainwsj = '/Users/iichr/Documents/Python/nlp/trainData/wsj_training/wsj_0001.txt'

# nltk.data.load('/Users/iichr/Documents/Python/nlp/trainData/wsj_untagged/names.family.txt', format="text")
# corpus_root = 'trainData/wsj_untagged'

"""
Names reading from given name files
"""

# name_read = nltk.corpus.reader.plaintext.PlaintextCorpusReader(path_names, '.*')
name_read = PlaintextCorpusReader(path_names, '.*')

names = []
for w in (name_read.words('names.male') + name_read.words('names.female') + name_read.words('names.family')):
    names.append(['PERSON', w])

# TESTING
# for name in names:
#     print(name, "\n")

"""
Extract training data
path -> [entities]

path = the path from which to do extraction of pre-named entities.
"""
# TODO add way to use all txt files in the specified path e.g loop through

def training_extract(path):


    enamex_pattern = re.compile(r'<ENAMEX.*?>.*?</ENAMEX>', re.ASCII)

    data = re.findall(enamex_pattern, nltk.data.load(path, format="text"))
    return data

# Test pattern to match all words
# enamex_pattern = re.compile(r'<ENAMEX.*?>.*?</ENAMEX>', re.ASCII)
# print(re.findall(enamex_pattern, nltk.data.load(path_trainwsj, format="text")))

print(training_extract(path_trainwsj))


# ####################
# Possibly useless   #
# ####################

"""
Sample name tagger from canvas (most likely useless)
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
