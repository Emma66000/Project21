import os
import re
from dataclasses import dataclass
from typing import Any, List, Set, Union
import nltk
import ir_datasets
import requests
from nltk.corpus import stopwords
from sqlitedict import SqliteDict
from collections import Counter

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def download_dataset(filename: str, force: bool = False) -> None:
    """Download a dataset to be used with ir_datasets.

    Args:
        filename: Name of the file to download.
        force (optional): Downloads a file and overwrites if already exists.
            Defaults to False.
    """
    filepath = os.path.expanduser(f"~/.ir_datasets/wapo/{filename}")
    if not force and os.path.isfile(filepath):
        return

    response = requests.get(f"https://gustav1.ux.uis.no/dat640/{filename}")
    if response.ok:
        print("File downloaded; saving to file...")
    with open(filepath, "wb") as f:
        f.write(response.content)

    print("First document:\n")
    print(next(ir_datasets.load("wapo/v2/trec-core-2018").docs_iter()))


def preprocess(doc: str) -> List[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ]


@dataclass
class Posting:
    doc_id: Union[str, int]
    payload: Any = None


class InvertedIndex(SqliteDict):
    def __init__(
        self,
        filename: str = "inverted_index.sqlite",
        fields: List[str] = ["title", "body"],
        new: bool = False,
    ) -> None:
        super().__init__(filename, flag="n" if new else "c")
        self.fields = fields
        self.index = {} if new else self
        """
        for field in self.fields:
            self.index[field]={}
        """
    def add_posting(self, term: str, field : str, doc_id: int, payload: Any=None) -> None:
        """Adds a document to the posting list of a term and field."""
        
        """
        if term not in  self.index[field].keys() :
             self.index[field][term]=list()
        self.index[field][term].append(Posting(doc_id,payload))
        
        """
        t=term+" "+field
        if t not in  self.index :
             self.index[t]=list()
        self.index[t].append(Posting(doc_id,payload))
        
    def get_postings(self, field: str, term: str) -> List[str]:
        """Fetches the posting list for a given field and term.

        Args:
            field: Field for which to get postings.
            term: Term for which to get postings.

        Returns:
            List of postings for the given term in the given field.
        """
        postings = list ()
        t=term+" "+field
        if t in self.index.keys():
            postings = self.index.get(t)
    
      
        return postings
    
    def get_term_frequency(self, field: str, term: str, doc_id: str) -> int:
        """Return the frequency of a given term in a document.

        Args:
            field: Index field.
            term: Term for which to find the count.
            doc_id: Document ID

        Returns:
            Term count in a document.
        """
        """
        cnt=0
        if term in self.index[field].keys() :
            for posting in self.index.get(field).get(term):
                if posting.doc_id ==doc_id :
                    cnt=posting.payload
         """   
         
        cnt=0
        t=term+" "+field
        if t in self.index.keys() :
            for posting in self.index.get(t, []):
                if posting.doc_id ==doc_id :
                    cnt=posting.payload
        return cnt
        

    def get_terms(self, field: str) -> Set[str]:
        """Returns all unique terms in the index.

        Args:
            field: Field for which to return the terms.

        Returns:
            Set of all terms in a given field.
        """
        
        #return set(self.index.get(field).keys())
        
        t=set()
        
        for key in self.index.keys():
            f=key.split()
            print(f)
            if f[1]==field :
                t.add(f[0])
        return t

    def __exit__(self, *exc_info):
        if self.flag == "n":
            self.update(self.index)
            self.commit()
            print("Index updated.")
        super().__exit__(*exc_info)


def index_collection(
    collection: str = "wapo/v2/trec-core-2018",
    filename: str = "inverted_index.sqlite",
    num_documents: int = 595037,
) -> None:
    """Builds an inverted index from a document collection.

    Note: WashingtonPost collection has 595037 documents. This might take a very
        long time to index on an average computer.


    Args:
        collection: Collection from ir_datasets.
        filename: Sqlite filename to save index to.
        num_documents: Number of documents to index.
    """
    dataset = ir_datasets.load(collection)
    with InvertedIndex(filename, new=True) as index:
       
        for i, doc in enumerate(dataset.docs_iter()):
            if i % (num_documents // 100) == 0:
                print(f"{round(100*(i/num_documents))}% indexed.")
            if i == num_documents:
                break

            #ajouter la term frequency, faire un counter pour ajouter que une fois
            
            d= preprocess(doc.body)
            cnt = Counter(d)
            for term in cnt.keys():
                index.add_posting(term, index.fields[1], doc.doc_id ,cnt.get(term))
                
            d=doc.title
            if d==None :
                d=""
            d=preprocess(d)
            cnt = Counter(d)
            for term in cnt.keys() :
                index.add_posting(term, index.fields[0], doc.doc_id ,cnt.get(term))
    #index.__exit__()
            
            
            
if __name__ == "__main__":
    download_dataset("WashingtonPost.v2.tar.gz")
    index_collection()  # total 595037 docs


