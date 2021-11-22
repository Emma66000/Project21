# Project21

## Indexing

Indexing is done from the file collection.tsv that can be downloaded from this link :
https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz

It is not included in the git repository as it is too big (almost 2Gb)

You can download the CAR collection from this link :
http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
The file is almost 16Gb

Unzip the folder in the data folder to get the following tree :
data/paragraphCorpus/*

## Baseline 

For the baseline, you can download files from:
https://github.com/daltonj/treccastweb/tree/master/2020/baselines
Then you can put them in the data folder like the following :
\data\baselines\y2_automatic_results_500.v1.0.run
...


## Qrels file

https://trec.nist.gov/data/cast/2020qrels.txt

This file have to be in the folder Scripts

## Trec eval tool

https://trec.nist.gov/trec_eval/
Unzip the folder in the Scripts folder.
Then compile the program : ```make``` 
This operation must be done under a Linux kernel (WSL/CGWIN).

## Requirements
python >= 3.7
Elasticsearch
Numpy
Sklearn
nltk
transformers
re
json


## Project files:
constants.py : File defining global variables
indexing.py : tools to idnex car and marco documents in elasticsearch index
Loading.py : Tools for file loading
main.py : Proejct execution file
Reranking_withBERT.py: File with reranking methods
rewriter: file containing method to rewrite queries


## How to run: 
First, install the requirements thanks to the following ```pip install -r requirements.txt``` 
Then you can run the main file via ```python main.py```
Make sure that all the different files are located in the good folder.
Also, you can change the behavior of the program with some options.
Run: python main.py 
with the follwoing argparse arguments: 
   * "--index", default=False, type=bool
   * --raw-trec-utterance", default=False, type=bool "If mentionned, uses raw queries instead of calculated rewritten queries"
   * "--auto-trec-utterance", default=False,  type=bool "If mentionned, uses raw queries instead of trec automatic rewritten queries"
   * "--no-rerank", default=False, type=bool,  "If mentionned, don't rerank files, just export results in trec_result.un if a result.txt file exists"
   * "--BERT?-rerank", default=True, type=bool, , "Toggles bert reranking")
   * "--print-rewritten", default=False, type=bool, If mentionned,write rewritten queries in a file")

