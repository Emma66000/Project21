# Project21

## Indexing

Indexing is done from the file collection.tsv that can be downloaded from this link :
https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz

It is not included in the git repository as it is too big (almost 2Gb)

You can download the CAR collection from this link :
http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
The file is almost 16Gb

## Baseline 

For the baseline, you can download files from:
https://github.com/daltonj/treccastweb/tree/master/2020/baselines


## Qrels file

https://trec.nist.gov/data/cast/2020qrels.txt

## Trec eval tool

https://trec.nist.gov/trec_eval/
Unzip the folder in the Scripts folder and run make in it. This operation must be done under a Linux kernel (WSL/CGWIN).

## Project files:
constants.py : File defining global variables
indexing.py : tools to idnex car and marco documents in elasticsearch index
Loading.py : Tools for file loading
main.py : Proejct execution file
Reranking_withBERT.py: File with reranking methods
rewriter: fiel containign method to rewrite queries


## How to run: 

Run: python main.py 
with the follwoing argparse arguments: 
    "--index", default=False, type=bool
    --raw-trec-utterance", default=False, type=bool "If mentionned, uses raw queries instead of calculated rewritten queries"
    "--auto-trec-utterance", default=False,  type=bool "If mentionned, uses raw queries instead of trec automatic rewritten queries"
    "--no-rerank", default=False, type=bool,  "If mentionned, don't rerank files, just export results if a result.txt file exists"
    "--BERT?-rerank", default=True, type=bool, , "Toggles bert reranking")
    "--print-rewritten", default=False, type=bool, If mentionned,write rewritten queries in a file")

