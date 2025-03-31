from langchain_community.document_transformers import LongContextReorder
from langchain_core.output_parsers import BaseOutputParser
from typing import List
from langchain.load import dumps, loads
# Output parser will split the LLM result into a list of queries

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]: #自定义parser只需要重载parse函数
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines
    
def rerank_documents(documents: list[list])->str:
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(documents)
    retrieve_context = ""
    for doc in reordered_docs:
        retrieve_context += doc.page_content
    return retrieve_context

def docs2str(documents: list[list])->str:
    retrieve_context = ""
    for doc in documents:
        retrieve_context += doc.page_content
    return retrieve_context

def fill_messages(content:str, role:str):
    return [{"role":role, "content":content}]

def reciprocal_rank_fusion(results: list[list], k=5):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for i, docs in enumerate(results):
        # Iterate through each document in the list, with its rank (position in the list)
        initial_weights = 1 if i>0 else 2  #用户初始查询权重更大
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k) * initial_weights

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    
    return reranked_results #[reranked_result[0] for reranked_result in reranked_results]