QRELS_FILE = "Scripts/2020qrels.txt"
INDEX_NAME = "index_project"
CAR_FILE = "data/paragraphCorpus/dedup.articles-paragraphs.cbor"
MARCO_FILE = "data/collection.tsv"
EXPORT_FILE = "data/trec_result.run"
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