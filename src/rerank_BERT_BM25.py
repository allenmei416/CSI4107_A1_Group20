from elasticsearch import Elasticsearch
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def run_queries_BERT(es, run_name, query_file, result_file, index_name, lambda_weight=0.5):
    output = open(result_file, "w")
    output.close()

    with open(query_file, "r") as infile:
        for i, line in enumerate(infile):
            query = json.loads(line)
            query_id = query['_id']
            query_string = query['text']

            response = es.search(
                index=index_name,
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
                bm25_score = doc['_score']
                retrieved_docs.append((doc_id, doc_title, doc_text, bm25_score))

            similarities = rerank_documents(query_string, retrieved_docs, lambda_weight)

            with open(result_file, "a") as results:
                rank = 1
                for doc_id, final_score in similarities:
                    result_string = f"{query_id} Q0 {doc_id} {rank} {final_score} {run_name}\n"
                    results.write(result_string)
                    rank += 1

    print(f"All queries retrieved and re-ranked. Results saved to {result_file}.")


def rerank_documents(query_string, retrieved_docs, lambda_weight):
    query_embedding = model.encode(query_string, convert_to_tensor=True)

    doc_texts = [doc_title + " " + doc_text for _, doc_title, doc_text, _ in retrieved_docs]
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings).squeeze().tolist()

    bm25_scores = np.array([bm25 for _, _, _, bm25 in retrieved_docs])

    min_bert = min(cosine_scores)
    max_bert = max(cosine_scores)
    bert_scores = [(score - min_bert) / (max_bert - min_bert) if max_bert > min_bert else 0 for score in cosine_scores]

    min_bm25 = np.min(bm25_scores)
    max_bm25 = np.max(bm25_scores)
    bm25_scores = [(score - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0 for score in bm25_scores]

    final_scores = [(doc_id, lambda_weight * bert + (1 - lambda_weight) * bm25) 
                    for (doc_id, _, _, _), bert, bm25 in zip(retrieved_docs, bert_scores, bm25_scores)]

    final_scores.sort(key=lambda x: x[1], reverse=True)

    return final_scores
