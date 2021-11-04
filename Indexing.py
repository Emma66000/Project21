import csv
import json
from typing import Callable, Dict, List, Set, Tuple
import math
import numpy as np
from elasticsearch import Elasticsearch



def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:
    """Indexes documents from file."""
    bulk_data = []
    cnt_passage =0

    file = open(filepath, encoding="utf8")
    read_tsv=csv.reader(file, delimiter="\t")
    print("Indexing as began...")
    
    
    for passage in read_tsv:
        cnt_passage+=1
        if cnt_passage==1:
            print(passage)
        
        print("Indexing in progress", cnt_passage*100/8841823 , "%")
        
        bulk_data.append(
            {"index": {"_index": index, "_id": passage[0:int(len(passage[0])/4)]}}
        )
        bulk_data.append({"body":passage[1]})
    
    
    es.bulk(index=index, body=bulk_data, refresh=True)
    print("Indexing Finished.")

def analyze_query(
    es: Elasticsearch, query: str, index: str = "_myindex"
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
        hits = (
            es.search(
                index=index,
                query={"match": {"body": t["token"]}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms

def load_queries(filepath: str) -> Dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.


    Take as query ID the value (on the same line) after `<num> Number: `, 
    and take as the query string the rest of the line after `<title> `. Omit
    newline characters.

    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.

    Returns:
        A dictionary with query IDs and corresponding query strings.
    """
    # TODO
    d={}
    key=""
    with open(filepath,'r', encoding="utf-8") as file :
        for line in file :
            line = json.loads(line)
            ln=line.strip().split(" ")
            if ln[0]=="<num>" :
                key=ln[-1]
            elif ln[0]=="<title>" :
                d[key]=" ".join(ln[1:])
            line = file.readline()
    file.close()
    
    return d

if __name__ == "__main__":
    index_name = "_myindex"
    es = Elasticsearch(timeout=120)

    index_documents("collection.tsv", es,index=index_name)

