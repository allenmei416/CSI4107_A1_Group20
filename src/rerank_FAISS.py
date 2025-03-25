import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch


model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
retrieved_docs = []  # Store document data

def run_queries_FAISS(es, run_name, query_file, result_file, index_name):

    output = open(result_file, "w")
    output.close()

    # read queries from file
    with open(query_file, "r") as infile:
        for i, line in enumerate(infile):
            query = json.loads(line)
            query_id = query['_id']
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

            index = None
            retrieved_docs = [] 

            doc_embeddings = []
            for doc in response["hits"]["hits"]:
                doc_id = doc['_id']
                doc_title = doc['_source']['title']
                doc_text = doc['_source']['text']
                doc_content = doc_title + " " + doc_text
                
                doc_embedding = model.encode(doc_content)
                retrieved_docs.append((doc_id, doc_title, doc_text))
                doc_embeddings.append(doc_embedding)
            
            doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
            
            index = faiss.IndexFlatL2(doc_embeddings.shape[1])
            index.add(doc_embeddings)

            # retrieve top 100 documents using FAISS
            similarities = search_faiss(retrieved_docs, index, query_string, top_k=100)

            # write results to file
            with open(result_file, "a") as results:
                rank = 1
                for doc_id, similarity in similarities:
                    result_string = f"{query_id} Q0 {doc_id} {rank} {similarity} {run_name}\n"
                    results.write(result_string)
                    rank += 1

    print(f"All queries retrieved and reranked. Results saved to {result_file}.")



def search_faiss(retrieved_docs, index, query_string, top_k=100):
    query_embedding = model.encode(query_string).reshape(1, -1)
    D, I = index.search(query_embedding, k=top_k)
    faiss_results = [(retrieved_docs[i][0], D[0][idx]) for idx, i in enumerate(I[0])]
    faiss_results.sort(key=lambda x: x[1])
    return faiss_results
