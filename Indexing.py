import csv
from elasticsearch import Elasticsearch
from trec_car.read_data import iter_paragraphs
from tqdm import tqdm 
from constants import *






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

def reset_index(es: Elasticsearch) -> None:
    """Clears index"""
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
