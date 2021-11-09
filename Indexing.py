import csv
import json
from typing import Callable, Dict, List, Set, Tuple
import math
import numpy as np
import pprint
from elasticsearch import Elasticsearch
from trec_car.read_data import iter_paragraphs
from threading import Thread
from tqdm import tqdm 
INDEX_NAME = "index_project"
CAR_FILE = "data/paragraphCorpus/dedup.articles-paragraphs.cbor"
MARCO_FILE = "data/collection.tsv"
INDEX_SETTINGS = {
    "mappings":{
        "properties":{
            "body": {
                "type":"text",
                "term_vector":"yes",
                "analyzer":"english"
            }
        }
    }
}

MARCO_COLLECTION_SIZE = 8841823
CAR_COLLECTION_SIZE = 29794697

def index_marco_documents(filepath: str, es: Elasticsearch, index: str) -> None:

    """Indexes documents from the file, this function assumes that, in the file documents are separated by \t

    Params:
        :filepath (str): path of the file to parse
        :es (ElasticSearch): elasticsearch object to use to index documents
        :index (str): index name to index documents

    """
    
    bulk_data = []
    cnt_passage=0
    with open(filepath, encoding="utf8") as docs:
        read_tsv=csv.reader(docs, delimiter="\t")

        print("Indexing as began...")
        for passage in tqdm(read_tsv, desc="MARCO indexing", total=MARCO_COLLECTION_SIZE):
            cnt_passage+=1
            
            bulk_data.append(
                {"index": {
                    "_index": index,
                    "_id":f"MARCO_{passage[0]}"}
                }
            )
            bulk_data.append({"body":passage[1]})
            
            if cnt_passage%(100000)==0 or cnt_passage == MARCO_COLLECTION_SIZE:
                es.bulk(index=index, doc_type="_doc", body=bulk_data, refresh=True)
                bulk_data=[]

    print("Indexing Marco Finished.")

def index_car_documents(filepath: str, es: Elasticsearch, index: str) -> None:

    """Indexes documents from the file, this function assumes that, in the file documents are separated by \t

    Params:
        :filepath (str): path of the file to parse
        :es (ElasticSearch): elasticsearch object to use to index documents
        :index (str): index name to index documents

    """
    
    bulk_data = []
    cnt_passage=0
    with open(filepath, mode="rb") as docs:
        iterator = iter_paragraphs(docs)

        print("Indexing as began...")
        for p in tqdm(iterator, desc="CAR indexing", total=CAR_COLLECTION_SIZE):
            cnt_passage+=1
            
            bulk_data.append(
                {"index": {
                    "_index": index,
                    "_id":f"CAR_{p.para_id}"}
                }
            )
            bulk_data.append({"body":p.get_text()})
            
            if cnt_passage%(100000)==0 or cnt_passage == CAR_COLLECTION_SIZE:
                es.bulk(index=index, doc_type="_doc", body=bulk_data, refresh=True)
                bulk_data=[]

    print("Indexing Car Finished.")

def analyze_query(
    es: Elasticsearch, query: str, index: str = "myindex"
) -> List[str]:
    """Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    """
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the
        # term.
        hits = es.search(index=index,query={"match": {"body": t["token"]}},_source=False,size=100,)["hits"]["hits"]
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms

def load_queries(filepath: str) -> Dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.


    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.

    Returns:
        A dictionary with query IDs and corresponding query strings.
    """
    # TODO
    d={}
    key=""
    with open(filepath,"r", encoding="utf-8") as file :
        file=json.load(file)
        for n in file:
            key=str(n['number'])+"_"
            for i in n['turn']:
                d[key+str(i['number'])]=i['manual_rewritten_utterance']
            key=""
    return d
def reset_index(es: Elasticsearch) -> None:
    """Clears index"""
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
if __name__ == "__main__":
    es = Elasticsearch(timeout=120)
    query={}
    query_terms=[]
    query=load_queries("data/2020_manual_evaluation_topics_v1.0.json")
    reset_index(es)
    index_marco_documents(MARCO_FILE, es,INDEX_NAME)
    index_car_documents(CAR_FILE, es, INDEX_NAME)
    query_terms=analyze_query(es, query['81_1'], INDEX_NAME) 
    print(query_terms)

