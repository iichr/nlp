## Named Entity Recognition NLP Project Report and Findings


We set about on the task of named entity recognition by defining methods to extract thedesired categories, namely Person, Location and Organisation, referred to simply as categorieshereinafter, from the training data provided.

In order to aid the recognition of names a list of honourifics in English is parsed [1](https://github.com/dariusk/corpora/blob/master/data/humans/englishHonorifics.json) , including but not limited to trivial titles such as Mr. and Mrs. In addition to that list, a corpus consisting of male, female and family names is also provided.

Having recognised that the locations extracted from training are not quite enough, we use the Ordnance Survey’s Names API [2](https://developer.os.uk/shop/places-networks/os-names-api.html), providing us with a list of 870,000 road names and 44,000 settlements in the United Kingdom.

Whilst this may not be exhaustive enough, we acknowledge that the data we are working with is predominantly centred on North America, thus owing to historical and cultural reasons, a proportion of the place names should have counterparts overseas.

-----

With the goal of improving recognition of organisations in particular, a method which takes all organisation endings in our training set and sorts them by their respective frequency in descending order (the most common ones) is implemented. We filter out proper names of organisations consisting of one word only. The top 10 in order are as follows:

1.  'Inc.'
2.  'Corp.'
3.  'Co.'
4.  'Ltd.'
5.  'Bank'
6.  'Group'
7.  'Association'
8.  'PLC'
9.  'University'
10. 'Committee'

Using a regular expression pattern we convert the extracted entities along with their categories to a form that would provide us with more data - the **part of speech tags** (*POS tags* hereinafter) in the given entity. A function is mapped to the list of extracted entities to convert it to a list of tuples, where the first entry is the category of the named entity, followed by the latter itself, in turn followed by a list of the POS tags for each separate part of the named entity composite, and finally a list ofjust the part of speech tags in the order they were observed. That information has been extractedby using `nltk.pos_tag`.

That list of tuples contains valuable information, which can be put into use to form a grammar. However it must first be brought into some order.

At first we settled on sorting how frequent a given part of speech tag pattern occurs in each category, and that formed the basis of our grammar. Certain patterns do occur more often than others, the initial sort identifies the top 5 most common, category notwithstanding, which are extracted in the file `filepospatterns_byfreq.txt`:

1. ('ORGANIZATION', ('NNP',))
2. ('LOCATION', ('NNP',))
3. ('ORGANIZATION', ('NNP', 'NNP'))
4. ('ORGANIZATION', ('NN',))
5. ('PERSON', ('NNP', ‘NNP'))

Those are rather vague and result in ambiguities, which correctness at the end is contingent on. It is to be noted that their sheer size, in our case 470 such unique patterns, of which 349 used as grammar rules, means that our grammar is quite specific, with overlaps between the categories and thus would fail to recognise some broader patterns. We obtained a list of the most common POS patterns by category, entitled `pospatterns.txt`. Those were fed into a grammar, which was passed on to a regular expression
parser, using the `nltk.RegexpParser()` method. However, this approach yielded tagged entities consisting of predominantly a single word - perhaps the result of trying to match a single POS tag pattern first such as NNP or NN, since they were located atop of the grammar, and are indeed the most frequent, as shown in the top 5 above. 

An attempt was made to mitigate this effect by sorting the tuples firstly by category and afterwards by the length of the POS patterns rather than their frequency, which fared better.

Three methods, corresponding to the three aforementioned categories are further used to carry out checks on which category an entity would fall under in the `cattagger()` method, where we also specify a parameter `backtrack1` which looks at the word just before an entity in question, with the hope of improving accuracy.

The test data is then loaded and extracted into a single string, encompassing all test files. This is then tokenised using `nltk.sent_tokenize()`. New lines “\n” are removed from the resulting listof sentences. Iterating one by one through the latter, the following sequence of procedures takesplace:

1.   `nltk.word_tokenize()` is applied to the sentence to split it into words

2.   the words are POS tagged using `nltk.pos_tag()`

3.   A chunk parser is applied to the POS tagged sentence

4.  The paths in the resulting tree are followed until we reach a leaf node, then we obtain it’s label and value, matching the former with the categories, thus only considering it if it falls within one of them.

5.  A list of named entities matching the patterns in the grammar is obtained. For each of them:

    (5)(i) we look for the word preceding the entity

    (5)(ii) the category tagger method is invoked, with both the entity and that word

    (5)(iii) the entity is assigned a tag from our set of 3, that is if it falls under it.

6. A list of all tagged entities is returned.

-----
The use of efficient data structures and optimal counting/sorting algorithms was part of the aims throughout the development process.

Our second revised approach to the grammar was not without its shortcomings, although faring better than the first one. A lot of entities were tagged as Organisations seemingly owing to the overlap of tagging patterns, as well as due to the fact that Organisations are the first entries in the grammar, since it is also ordered by category. 
