from elasticsearch import Elasticsearch
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def run_queries_BERT(es, run_name, query_file, result_file, index_name):
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


            # get new ranking scores using embeddings and cosine similarity
            similarities = rerank_documents(query_string, retrieved_docs)

            # write results to file
            with open(result_file, "a") as results:
                rank = 1
                for doc_id, similarity in similarities:
                    result_string = f"{id} Q0 {doc_id} {rank} {similarity} {run_name}\n"
                    results.write(result_string)
                    rank += 1

    print(f"All queries retrieved and reranked. Results saved to {result_file}.")


def rerank_documents(query_string, retrieved_docs):
    # generate embeddings
    query_embedding = model.encode(query_string, convert_to_tensor=True)
    doc_embeddings = []
    for doc_id, doc_title, doc_text in retrieved_docs:
        doc_content = doc_title + " " + doc_text
        doc_embedding = model.encode(doc_content, convert_to_tensor=True)
        doc_embeddings.append((doc_id, doc_embedding))

    # calculate cosine similarity
    similarities = []
    for doc_id, doc_embedding in doc_embeddings:
        similarity = util.pytorch_cos_sim(query_embedding, doc_embedding)[0][0].item()
        similarities.append((doc_id, similarity))

    # sort similarity scores
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities

