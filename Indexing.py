import csv
import json
from typing import Callable, Dict, List, Set, Tuple
import math
import numpy as np
import pprint
import random
import time
from sklearn.linear_model import LinearRegression
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

FEATURES_QUERY = [
    "query_length",
    "query_sum_idf",
    "query_max_idf",
    "query_avg_idf",
]
FEATURES_DOC = ["doc_length_body"]
FEATURES_QUERY_DOC = [
    "unique_query_terms_in_body",
    "sum_TF_body",
    "max_TF_body",
    "avg_TF_body",
]
FIELDS = ["body"]

COLLECTION_SIZE = 8841823


def get_doc_term_freqs(
    es: Elasticsearch, doc_id: str, index: str = INDEX_NAME
) -> Dict[str, int]:
    """Gets the term frequencies of a field of an indexed document.

    Args:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    """
    field="body"
    tv = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    )
    if tv["_id"] != doc_id:
        return None
    if field not in tv["term_vectors"]:
        return None
    term_freqs = {}
    for term, term_stat in tv["term_vectors"][field]["terms"].items():
        term_freqs[term] = term_stat["term_freq"]
    return term_freqs


def extract_query_features(
    query_terms: List[str], es: Elasticsearch, index: str = INDEX_NAME
) -> Dict[str, float]:
    """Extracts features of a query.

        Args:
            query_terms: List of analyzed query terms.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.
        Returns:
            Dictionary with keys 'query_length', 'query_sum_idf',
                'query_max_idf', and 'query_avg_idf'.
    """
    # TODO

    d={}
    d['query_length']=len(query_terms)
    d['query_sum_idf']=0
    d['query_max_idf']=0
    d['query_avg_idf']=0
    
    query_idf=list()

    for term in query_terms:
        hits = (es.search(index=index,query={"match": {"body": term}},_source=False,size=1,).get("hits", {}).get("hits", {}))
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        tv = es.termvectors(index=index, id=doc_id, fields=["body"], term_statistics=True)
        df=tv["term_vectors"]["body"]['terms'].get(term,{}).get('doc_freq',0)
        N=es.count(index=index)['count']
        if df>0 :
            idf=np.log(N/df)
        else :
            idf=0
        query_idf.append(idf)

    #if len(query_idf)>0:
    d['query_sum_idf']=sum(query_idf)
    if query_idf :
        d['query_max_idf']=max(query_idf)
    if query_terms :
        d['query_avg_idf']=d['query_sum_idf']/len(query_terms)
    

    
    return d


def extract_doc_features(
    doc_id: str, es: Elasticsearch, index: str = INDEX_NAME
) -> Dict[str, float]:
    """Extracts features of a document.

        Args:
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            Dictionary with keys 'doc_length_body'.
    """
    # TODO
    d={}
    d["doc_length_body"]=0

    term_freqs=get_doc_term_freqs(es, doc_id, index)
    if term_freqs :
        d["doc_length_body"]=sum(term_freqs.values())

    return d


def extract_query_doc_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = INDEX_NAME,
) -> Dict[str, float]:
    """Extracts features of a query and document pair.

        Args:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            Dictionary with keys
                'unique_query_terms_in_body', 'sum_TF_body',
                , 'max_TF_body', 'avg_TF_body'. 
    """
    # TODO
    d={}
    d['unique_query_terms_in_body']=0
    d['sum_TF_body']=0
    d['max_TF_body']=0
    d['avg_TF_body']=0
    
    TF = {}
    TF['body']=list()

    field="body"
    if doc_id :
        
        for term in query_terms:
            tv = es.termvectors(index=index, doc_type="_doc", id=doc_id, fields=field, term_statistics=True)
            if tv["_id"] != doc_id:
                continue
            if field not in tv["term_vectors"]:
                continue
            if term not in tv["term_vectors"][field]['terms']:
                continue

            TF[field].append(tv["term_vectors"][field]['terms'][term]["term_freq"])

            if tv["term_vectors"][field]['terms'][term]["term_freq"]==1:
                if field=="body":
                    d['unique_query_terms_in_body']+=1
                        
        
        
        if TF['body'] :  
            d['sum_TF_body']=sum(TF['body'])
            d['max_TF_body']=max(TF['body'])
            d['avg_TF_body']=sum(TF['body'])/len(query_terms)



    return d


def extract_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = INDEX_NAME,
) -> List[float]:
    """Extracts query features, document features and query-document features
        of a query and document pair.

        Args:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            List of extracted feature values in a fixed order.
    """
    query_features = extract_query_features(query_terms, es, index=index)
    feature_vect = [query_features[f] for f in FEATURES_QUERY]
    #print(feature_vect)
    doc_features = extract_doc_features(doc_id, es, index=index)
    feature_vect.extend([doc_features[f] for f in FEATURES_DOC])
    #print(feature_vect)
    query_doc_features = extract_query_doc_features(
        query_terms, doc_id, es, index=index
    )
    feature_vect.extend([query_doc_features[f] for f in FEATURES_QUERY_DOC])

    return feature_vect


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
        
        
class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor.
        """
        # TODO
        self.regressor = LinearRegression()

    def _train(self, X: List[List[float]], y: List[float]) -> None:
        """Trains an LTR model.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(
        self, ft: List[List[str]], doc_ids: List[str]
    ) -> List[Tuple[str, int]]:
        """Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def get_rankings(
    ltr: PointWiseLTRModel,
    query_ids: List[str],
    all_queries: Dict[str, str],
    es: Elasticsearch,
    index: str,
    rerank: bool = False,
) -> Dict[str, List[str]]:
    """Generate rankings for each of the test query IDs.

    Args:
        ltr:  A trained PointWiseLTRModel instance.
        query_ids: List of query IDs.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    """

    test_rankings = {}
    for i, query_id in enumerate(query_ids):
        print(
            "Processing query {}/{} ID {}".format(
                i + 1, len(query_ids), query_id
            )
        )
        # First-pass retrieval
        query_terms = analyze_query(
            es, all_queries[query_id], index=index
        )
        if len(query_terms) == 0:
            print(
                "WARNING: query {} is empty after analysis; ignoring".format(
                    query_id
                )
            )
            continue
        hits = es.search(
            index=index, q=" ".join(query_terms), _source=True, size=100
        )["hits"]["hits"]
        test_rankings[query_id] = [hit["_id"] for hit in hits]

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            # TODO
            X=list()
            Y=list()
            for d_id in test_rankings[query_id]:
                X.append(extract_features(query_terms, d_id, es, index))
                
            for res in ltr.rank(X,test_rankings[query_id]):
                if res[1]>0:
                    Y.append(res[0])
            test_rankings[query_id]=Y
            #print(query_id,test_rankings)

    return test_rankings

def prepare_ltr_training_data(
    query_ids: List[str],
    all_queries: Dict[str, str],
    all_qrels: Dict[str, List[str]],
    es: Elasticsearch,
    index: str,
) -> Tuple[List[List[float]], List[int]]:
    """Prepares feature vectors and labels for query and document pairs found
    in the training data.

        Args:
            query_ids: List of query IDs.
            all_queries: Dictionary containing all queries.
            all_qrels: Dictionary with keys as query ID and values as list of
                relevant documents.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            X: List of feature vectors extracted for each pair of query and
                retrieved or relevant document.
            y: List of corresponding labels., 0 if not relevant and 1 if relevant from query_list
    """
    # TODO
    #document from all_qrels
    #search avec es to find top 100 doc
    #extract features with query

    X=list()
    y=list()

    for query in query_ids :
        print(query)
        query_terms = analyze_query(es, all_queries[query],index)
        
        for d_id in all_qrels.get(query,{}):
            X.append(extract_features(query_terms, d_id, es, index))
            y.append(1)

        res = es.search(index=index, q=" ".join(query_terms), size=100)["hits"]["hits"]
        for doc in res:
            if doc["_id"] not in all_qrels.get(query,{}):
                X.append(extract_features(query_terms,doc["_id"], es, index))
                y.append(0)

                
    return X, y

def train_test_split(queries):
    random.seed(a=1234567)
    query_ids = sorted(list(queries.keys()))
    random.shuffle(query_ids)
    train_size = int(len(query_ids) * 0.8)
    return query_ids[:train_size], query_ids[train_size:][-100:]
    
    

def training_data(es, queries, qrels):
    # Prepare training data with labels for learning-to-rank
    train, _ = train_test_split(queries)
    return prepare_ltr_training_data(
        train[:800], queries, qrels, es, index=INDEX_NAME
    )
    
    

def trained_ltr_model(training_data):
    X_train, y_train = training_data
    # Instantiate PointWiseLTRModel.
    ltr = PointWiseLTRModel()
    ltr._train(X_train, y_train)
    return ltr

def load_qrels(filepath: str) -> Dict[str, List[str]]:
    """Loads query relevance judgments from a file.


    Example :
        
    81_2 Q0 MARCO_1900267 54 0.0004690346249844879 PASBERT_manual_rewritten
    81_2 Q0 MARCO_6519779 55 0.0004508823622018099 PASBERT_manual_rewritten
    81_2 Q0 MARCO_7948855 56 0.0004076514160260558 PASBERT_manual_rewritten
    81_2 Q0 MARCO_3942604 57 0.00038496305933222175 PASBERT_manual_rewritten
    81_2 Q0 CAR_aee7be1029e24bf71f3a08d4a5593938bda63769 77 0.00020508274610619992 PASBERT_manual_rewritten
    
    Args:
        filepath: String (constructed using os.path) of the filepath to a
            file with queries.

    Returns:
        A dictionary with query IDs and a corresponding list of document IDs
            for documents judged relevant to the query.
    """
    # TODO
    d={}
    key=""
    cnt=0
    with open(filepath,'r', encoding="utf-8") as file :
        line = file.readline()
        
        while line:  
            print(cnt)
            ln=line.strip().replace('"', "").split(" ")
            if ln[2][0:3]!="CAR" : 
                key = ln[0]
                if key not in d.keys():
                    d[key]=list()
                d[key].append(ln[2])
            line = file.readline()
            cnt+=1
    file.close()
    return d

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
if __name__ == "__main__":
    es = Elasticsearch(timeout=120)
    query={}
    query_terms=[]
    query=load_queries("data/2020_manual_evaluation_topics_v1.0.json")
    reset_index(es)
    index_marco_documents(MARCO_FILE, es,INDEX_NAME)
    index_car_documents(CAR_FILE, es, INDEX_NAME)
    qrels=load_qrels("data/baselines/y2_manual_results_500.v1.0.run")
    #reset_index(es)
    #index_documents("data/collection.tsv", es,index=INDEX_NAME)
    #print(es.termvectors(index=INDEX_NAME, id='1'))
    query_terms=analyze_query(es, query['81_1'], INDEX_NAME) 
   # print(query_terms)
   
    _, test = train_test_split(query)
    rankings_ltr = get_rankings(trained_ltr_model(training_data(es,query,qrels)), test, query, es, index=INDEX_NAME, rerank=True)

