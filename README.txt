ID:1527549

The main file is ner.py

When run, it should automatically perform NER on the newly released data*
*that is if you have the same username and folder structure as I do (highly unlikely...)


The following method is to be called for other runs:

------
namedentityrecognition(taggedpath,untaggedpath)
------
where:
	taggedpath = the path with the tagged (training) data.
	untaggedpath = the path with the untagged (test) data, which to perform NER on.


More information on the code can be found in the docstrings in the ner.py file.
For analysis and descripiton of the methods used, refer to the file entitled [Assignment1.pdf],
which contains a report.
