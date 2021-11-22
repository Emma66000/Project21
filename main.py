import logging
from constants import EXPORT_FILE, CAR_FILE, INDEX_NAME, QRELS_FILE
import json
from typing import Dict
from elasticsearch import Elasticsearch
from Indexing import *
from Reranking_withBERT import *
from Loading import *
from os import path, truncate
from argparse import ArgumentParser
from rewriter import Qrewriter

log = logging.getLogger("Project")
log.setLevel(logging.DEBUG)
handler = logging.FileHandler('indexing.log','w','utf-8')
log.addHandler(handler)



def export_trec_result(ranking):
    """Creates a trec_result.run file that contains the ranking in the TREC result file format.
    <query-id> Q0 <document-id> <rank> <score> STANDARD

    Args:
        ranking (dict): dictionnary of queries and their ranking
    """
    with open(EXPORT_FILE, "w") as f:
        lines = []
        for query_id, ranks in ranking.items():
            for i, r in enumerate(ranks):
                content_line = [query_id, "Q0", r[0],i,  -1*r[1], "STANDARD"]
                lines.append(" ".join([str(f) for f in content_line]))
        f.write("\n".join(lines))
            
def rewrite_queries(es, queries: Dict[str, str],load_ti: Dict[str, str]) -> Dict[str, str]:
    rw = Qrewriter(queries,load_ti)

    d = {}
    for query_id in queries:
        d[query_id] = rw.rewrite_query_with_context(es, query_id)
    
    return d



if __name__ == "__main__":
    parser = ArgumentParser("TREC CAST 2020", description="Conversationnal assistance project")
    parser.add_argument("--index", default=False, type=bool, help="If mentionned, index the file collection", const=True, nargs="?")
    parser.add_argument("--raw-trec-utterance", default=False, nargs='?', type=bool, const=True, help="If mentionned, uses raw queries instead of calculated rewritten queries")
    parser.add_argument("--auto-trec-utterance", default=False, nargs='?', type=bool, const=True, help="If mentionned, uses raw queries instead of trec automatic rewritten queries")
    parser.add_argument("--no-rerank", default=False, nargs='?', type=bool, const=True, help="If mentionned, don't rerank files, just export results if a result.txt file exists")
    parser.add_argument("--BERT", default=False, nargs='?', type=bool, const=True, help="Toggles BERT re-ranking")
    parser.add_argument("--print-rewritten", default=False, nargs='?', type=bool, const=True, help="If mentionned,write rewritten queries in a file")
    args = parser.parse_args()
    es = Elasticsearch(timeout=120)
    if args.index:
        reset_index(es)
        index_marco_documents(MARCO_FILE, es,INDEX_NAME)
        index_car_documents(CAR_FILE, es, INDEX_NAME)
    if not args.no_rerank:
        raw_query=load_queries("data/2020_manual_evaluation_topics_v1.0.json", args.auto_trec_utterance)
        
        if not args.raw_trec_utterance:
            titles=load_titles("data/automatic_evaluation_topics_annotated_v1.1.json")
            re_query=rewrite_queries(es, raw_query, titles)
        else :
            re_query=raw_query

        log.info("rewritting query complete")
        if args.print_rewritten:
            with open("rewritten_query.txt", 'w') as f:
                json.dump(re_query, f, indent=2)
                
        qrels=load_qrels(QRELS_FILE)
        
        log.info("Analyze query complete")
        _, test = train_test_split(re_query)
        log.info("Test train complete")
        trained_data = training_data(es,re_query,qrels)
        log.info("Train data complete")
        model = trained_ltr_model(trained_data)
        log.info("Train model complete")
        rankings_ltr = get_rankings(model, test, re_query, es, index=INDEX_NAME, rerank=False)

        if args.BERT:
            print('BERT TIME')
            rankings_ltr = bert_rerank(re_query, es, rankings_ltr, index=INDEX_NAME)


        with open("result.txt", 'w') as f:
            json.dump(rankings_ltr, f, indent=2)
        log.info("Ranking complete")
        log.info("Evaluating")
        evaluate = test_mean_rr(es,INDEX_NAME,test,trained_data,model,rankings_ltr,re_query,qrels)
        with open("eval.txt", 'w') as f:
            json.dump(evaluate, f, indent=2)
        log.info(f"Evaluation complete {evaluate}")
        print(evaluate)
    else:
        log.info("Ignoring rerank stage")

    if path.exists("result.txt"):
        with open("result.txt", "r") as f:
            rankings_ltr_run = json.load(f)
            export_trec_result(rankings_ltr_run)
        log.info("Results export complete")
    else:
        log.warning("no result file result.txt found")
    
    
    