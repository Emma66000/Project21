from typing import Callable, Dict, List, Set, Tuple
from elasticsearch import Elasticsearch

from math import log
from constants import TOTAL_COLLECTION_SIZE, INDEX_NAME

from Loading import load_queries, load_titles




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
    tokens = es.indices.analyze(index=INDEX_NAME, body={"text": query})["tokens"]
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



class Qrewriter():
    """Class for query rewriting"""
    def __init__(self, queries: Dict[str, str],titles: Dict[str, str]):
        self.titles = titles
        self.queries = queries
        self.context= dict()
        self.index_num_docs = TOTAL_COLLECTION_SIZE
    

    def title_rewrite(self):
        """ Appends turn title to query
        """
        d = self.queries
        for i in self.titles.keys():
            for q in self.queries.keys():
                if i in q :
                    d[q]=self.queries[q]+" "+self.titles[i]
        return d
    
    def __process_context(self,es, query_id):
        """ Uses inverse document score ( just idf, not tf) to estimate context,
        stores context for each turn. 
        """
        turn_id = query_id.split('_')[0]
        kept_term_size = 3
        distance_factor = 0.8 # Factor than scores down context from older queries 

        termscores ={}
        query_terms = analyze_query(es, self.queries[query_id], index=INDEX_NAME)
        
        for t in query_terms:
            hits = es.search(index=INDEX_NAME,query={"match": {"body": t}},_source=False)["hits"]["hits"]
            doc_id = hits[0]["_id"] if len(hits) > 0 else None
            if doc_id is None:
                continue
            tv = es.termvectors(
            index=INDEX_NAME, id=hits[0]['_id'], fields='body', term_statistics=True)

            df=tv["term_vectors"]["body"]['terms'][t]['doc_freq']
            ttf=tv["term_vectors"]["body"]['terms'][t]['ttf']

            idf = log(self.index_num_docs/df) # Using idf, seems like it works somewhat
            termscores[t] = idf
        
        if turn_id not in self.context.keys():
            self.context[turn_id] = sorted(termscores.items(), key =lambda x: x[1],reverse=True)[:kept_term_size]
        else:
            out_terms = list(termscores.items())
            for context_tup in self.context[turn_id]:
                if context_tup not in out_terms:
                    c_term, c_score= context_tup
                    if c_term not in query_terms:
                        c_score = distance_factor*context_tup[1]
                    
                    out_terms.append((c_term, c_score))
            self.context[turn_id] = sorted(out_terms, key =lambda x: x[1],reverse=True)[:kept_term_size]
    
    def reset_context(self):
        self.context = dict()

                    
    def rewrite_query_with_context(self,es, query_id):
        turn_id = query_id.split('_')[0]
        if turn_id in self.context:
            out = self.queries[query_id] +' '+ ' '.join(x[0] if x[0] not in self.queries[query_id] else '' for x in self.context[turn_id] )
        else:
            out = self.queries[query_id]
        self.__process_context(es, query_id)
        return out
            

def main():
    es = Elasticsearch(timeout= 120)
    index = INDEX_NAME

    auto_trec_rewrite = False
    queries = load_queries("data/2020_manual_evaluation_topics_v1.0.json",auto_trec_rewrite)
    titles=load_titles("data/automatic_evaluation_topics_annotated_v1.1.json")
    qr_test = Qrewriter(queries, titles)

    for query_id in queries:
        print(qr_test.rewrite_query_with_context(es, query_id))
        

if __name__ == '__main__':
    main()

    


    