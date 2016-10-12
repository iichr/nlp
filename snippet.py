from collections import Counter
import re
import string

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('/home/students/cii549/work/Downloads/big.txt').read()))

# PROBABILITY OF A WORD OCCURING
def prob(word):
        return WORDS[word]/sum(WORDS.values())

# ONE CHARACTER EDITS
def edits1(word):
        # a to z and append, same with prefix -> check if in WORDS
        allletters = string.ascii_lowercase
        l1 = []

        for i in allletters:
            newword = word+i
            if newword in WORDS:
                l1.append(newword)

        for i in allletters:
            newword1 = i+word
            if newword1 in WORDS:
                l1.append(newword1)
        # print(l1)

        
        for i in allletters:
                z=0
                while z < len(word):
                        newword2 = word.replace(word[z],i)
                        # word[z]=word[k]
                        if newword2 in WORDS:
                                l1.append(newword2)
                                z += 1
                                print(l1)
        
            
        # loop over all letter positions, one by one: word[0:len(word)]
        # +1 and -1
        # check if in WORDS: WORDS.get(newword)
        # initialise list: l1[]
        # add to list if so: l1.append(newword)
