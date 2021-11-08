import csv
import json
from typing import Callable, Dict, List, Set, Tuple
import math
import numpy as np
import pprint
from elasticsearch import Elasticsearch

INDEX_NAME = "myindex_3"
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

def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:

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
        for passage in read_tsv:
            cnt_passage+=1
            if cnt_passage % int(8841823/100) == 0:
                print("Indexing in progress", round(cnt_passage*100/8841823) , "%")
            bulk_data.append(
                {"index": {
                    "_index": index,
                    "_id":passage[0]}
                }
            )
            bulk_data.append({"body":passage[1]})
            
            if cnt_passage%(1000)==0 :
                es.bulk(index=index, doc_type="_doc", body=bulk_data, refresh=True)
                bulk_data=[]

    print("Indexing Finished.")

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
    print("query",query)
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    print(tokens)
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the
        # term.
        print(t['token'])
        hits = es.search(index=index,query={"match": {"body": t["token"]}},_source=False,size=1,)
        print(hits)
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
    index_documents("data/collection.tsv", es,index=INDEX_NAME)
    print(es.termvectors(index=INDEX_NAME, id='1'))
    # query_terms=analyze_query(es, query['81_1'], INDEX_NAME) 
    # print(query_terms)

