from elasticsearch import Elasticsearch
from constants import INDEX_NAME, FEATURES_DOC, FEATURES_QUERY_DOC, FEATURES_QUERY
from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
bert_model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")


class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor.
        """
        # self.regressor = LinearRegression()
        self.regressor = MLPRegressor()

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
    if tv["_id"] != doc_id or not tv['found']:
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
            if tv["_id"] != doc_id or not tv['found']:
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
        print(json.dumps(hits))

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            test_rankings[query_id] = [hit["_id"] for hit in hits]

            X=list()
            Y=list()
            for d_id in test_rankings[query_id]:
                print("get rankings",d_id)
                X.append(extract_features(query_terms, d_id, es, index))
                
            for res in ltr.rank(X,test_rankings[query_id]):
                print(res)
                if res[1]>0:
                    Y.append(res)
            test_rankings[query_id]=Y
            #print(query_id,test_rankings)
        else:
            test_rankings[query_id] = [(hit["_id"], hit['_score']) for hit in hits]


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
    #document from all_qrels
    #search avec es to find top 100 doc
    #extract features with query

    X=list()
    y=list()

    for query in query_ids :
        print(query)
        #print(query)
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
    print("train test split did")
    return prepare_ltr_training_data(
        train, queries, qrels, es, index=INDEX_NAME
    )
    
    

def trained_ltr_model(training_data):
    X_train, y_train = training_data
    # Instantiate PointWiseLTRModel.
    ltr = PointWiseLTRModel()
    ltr._train(X_train, y_train)
    return ltr


def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: List[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of document
            IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean is
            computed over.

    Returns:
        Mean evaluation measure (float).
    """
    sum_score = 0

    if len(system_rankings)!=0:
        for query_id, system_ranking in system_rankings.items():
            sum_score += eval_function(system_ranking, ground_truths.get(query_id,[]))
        return sum_score / len(system_rankings)
    else :
        return 0

def test_mean_rr(es : Elasticsearch ,index:str ,test,trained_data,model,rankings_ltr,queries,qrels):

    """test the mean evaluation measure over a set of queries and compare it with the mean evaluation after re ranking
    """
    rankings_first_pass = get_rankings(None, test, queries, es, index=INDEX_NAME, rerank=False)
    mrr_first_pass = get_mean_eval_measure(rankings_first_pass, qrels, get_reciprocal_rank)

    d={}
    for query_id in rankings_ltr.keys():
        if query_id not in d:
            d[query_id]=list()
        for doc in rankings_ltr.get(query_id,[]):
            d[query_id].append(doc[0])
    print(d)
    mrr_ltr = get_mean_eval_measure(d, qrels, get_reciprocal_rank)
    return (mrr_first_pass, mrr_ltr - mrr_first_pass)

def load_query_bert(re_query):
    d={}
    for query_id in re_query.keys():
        if query_id not in d:
            d[query_id]=""
        d[query_id]="query: "+re_query[query_id]

    return d

def load_querydoc_bert(re_query,es,rankings,index):
    d={}
    query_bert = load_query_bert(re_query)
    for query_id in rankings.keys():
        if query_id not in d:
            d[query_id]={}
        d[query_id]["query"]=query_bert[query_id]
        d[query_id]["document"]=[]
        for doc_id in rankings.get(query_id):   
            res=es.get(index,doc_id)["_source"]["body"]
            d[query_id]["document"].append((doc_id,"passage: "+res))
    return d



def bert_rerank(re_query,es,rankings,index):
    data = load_querydoc_bert(re_query, es, rankings, index)
    reranking = {}
    for q_id in data.keys(): 
        q_string = data[q_id]["query"]
        d_strings = [x[1][:min(700,len(x[1]))] for x in  data[q_id]["document"]]
        d_ids = [x[0] for x in  data[q_id]["document"]]
        bert_input = tokenizer(text = [q_string]*len(d_strings), text_pair=d_strings, return_tensors='pt', padding = True)
        loss = bert_model(**bert_input).logits[:,0]
        scores = [(q, s.item()) for q,s in zip(d_ids, loss)]
        rankings = list(sorted(scores, key = lambda x: x[1]))
        reranking[q_id] = [[rk[0],rk[1]] for rk in rankings]

    return reranking


