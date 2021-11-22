from nltk.corpus import stopwords
from typing import Dict, List
import nltk
import json
import re

# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
STOPWORDS = set(stopwords.words("english"))

def load_titles(filepath: str) -> Dict[str, str]:
    """Loads the titles (Preprocessed first query+Title of the query), context of each query, from a file.
    """
    
    
    d={}
    new={}
    key=""
    with open(filepath,"r", encoding="utf-8") as file :
        file=json.load(file)
        for n in file:

            key=str(n['number'])
            words = f"{n.get('title', '')} {n['turn'][0]['raw_utterance']}".lower()
            words= re.sub(r"[^\w]|_", " ", words).split()
            
            new[key]=[]
            
            d[key]=nltk.pos_tag(words)
            for p in d[key]:
                if p[1] in ['NN','JJ','NNS','JJS','VBG']:
                    new[key].append(p[0]) 
                    
            d[key] = []        
            for term in new[key] :
                if term not in STOPWORDS and term not in d[key]:
                    d[key].append(term)     
            d[key]=" ".join(d[key])
            
    return d

def load_qrels(filepath: str) -> Dict[str, List[str]]:
    """Loads query relevance judgments from a file.


    Example :
        
    81_1 0 CAR_3add84966af079ed84e8b2fc412ad1dc27800127 1
    81_1 0 CAR_5fa30140b395d7fead223e2bca8cc9b608bb51b4 0
    81_1 0 CAR_aa2504ce1af15bace5d96daecf4ff491ffd39ae7 1
    81_1 0 CAR_c35739b96d529a5dd18c97a95154670f7416c9ef 0
    81_1 0 MARCO_1104225 2
    
    Returns : A dictionnary with query_turn as keys and list of Id of relevant document for this query_turn
    """
    
    d={}
    with open(filepath,'r', encoding="utf-8") as file :
        line = file.readline()
        
        while line:  
            ln=line.strip().split(" ") 
            key = ln[0]
            
            if key not in d.keys():
                d[key]=list()
                
            if int(ln[3])>1:
                d[key].append(ln[2])
                
            line = file.readline()
            
    return d

def load_queries(filepath: str, auto_trec_utterance: bool=False) -> Dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.

    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.
        
        auto_trec_utterance: boolean that indicate if we want to directly load the automatic rewritten queries of trec to test
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
                if auto_trec_utterance:
                    d[key+str(i['number'])]=i['automatic_rewritten_utterance']
                else :
                    d[key+str(i['number'])]=i['raw_utterance']
                    
            key=""
    return d
