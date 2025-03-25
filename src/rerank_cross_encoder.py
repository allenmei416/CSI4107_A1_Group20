from elasticsearch import Elasticsearch
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Cross-Encoder model and tokenizer
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def run_queries_cross_encoder(es, run_name, query_file, result_file, index_name):
    # create output file
    output = open(result_file, "w")
    output.close()

    # read queries from file
    with open(query_file, "r") as infile:
        for i, line in enumerate(infile):
            query = json.loads(line)
            id = query['_id']
            query_string = query['text']

            # run each query
            response = es.search(
                index=index_name,
                # query can be changed to 'match' for title or text only 
                query={
                   "multi_match": {
                      "query": query_string,
                      "fields": ["title", "text"] 
                   }
                },
                size=100
            )
            
            retrieved_docs = []
            for doc in response["hits"]["hits"]:
                doc_id = doc['_id']
                doc_title = doc['_source']['title']
                doc_text = doc['_source']['text']
                retrieved_docs.append((doc_id, doc_title, doc_text))
 
            # Re-rank documents using cross-encoder
            reranked_results = rerank_with_cross_encoder(query_string, retrieved_docs)
            
            # Write results
            with open(result_file, "a") as results:
                for rank, (doc_id, score) in enumerate(reranked_results, start=1):
                    result_string = f"{id} Q0 {doc_id} {rank} {score} {run_name}\n"
                    results.write(result_string)
    
    print(f"All queries retrieved and reranked. Results saved to {result_file}.")

def rerank_with_cross_encoder(query, retrieved_docs):
    doc_texts = [doc[1] for doc in retrieved_docs]
    
    # Tokenize query-document pairs
    inputs = tokenizer([query] * len(doc_texts), doc_texts, padding=True, truncation=True, return_tensors="pt")
    
    # Compute relevance scores
    with torch.no_grad():
        scores = cross_encoder_model(**inputs).logits.squeeze().tolist()
    
    # Sort documents by relevance score (higher is better)
    reranked_docs = sorted(zip([doc[0] for doc in retrieved_docs], scores), key=lambda x: x[1], reverse=True)
    return reranked_docs