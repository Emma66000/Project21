import csv
from typing import Callable, Dict, List, Set, Tuple
import math
import numpy as np
from elasticsearch import Elasticsearch



def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:
    """Indexes documents from file."""
    bulk_data = []
    cnt_passage =0

    file = open(filepath)
    read_tsv=csv.reader(file, delimiter="\t")
    print("Indexing as began...")
    
    for passage in read_tsv:
        cnt_passage+=1
        print("Indexing in progress", cnt_passage)
        bulk_data.append(
            {"index": {"_index": index, "_id": passage[0]}}
        )
        bulk_data.append(passage[1:])
    es.bulk(index=index, body=bulk_data, refresh=True)
    print("Indexing Finished.")


if __name__ == "__main__":
    index_name = "_myindex"
    es = Elasticsearch(timeout=120)

    index_documents("collection.tsv", es,index=index_name)

