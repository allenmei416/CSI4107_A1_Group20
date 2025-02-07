# Step 3. [10 points] Retrieval and Ranking:  Use the inverted index (from step 2) to find the 
# limited set of documents that contain at least one of the query words. Compute the cosine 
# similarity scores between a query and each document. 

# Input: One query and the Inverted Index (from Step2)

# Output: Similarity values between the query and each of the documents. Rank the documents in decreasing order of similarity scores.

from elasticsearch import Elasticsearch
import json

def run_queries(es, run_name, query_file, result_file, index_name):
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
                query = {
                   "multi_match": {
                      "query": query_string,
                      "fields": ["title", "text"] 
                   }
                },
                size=100
            )
            
            # get normalized scores
            normalized_scores = normalize_scores(response)
            
            # write results to file
            results = open(result_file, "a")
            rank = 0
            score_index = 0
            
            for doc in response["hits"]["hits"]:
                doc_id = doc['_id']
                score = normalized_scores[score_index]
                rank += 1
                score_index += 1

                result_string = f"{id} Q0 {doc_id} {rank} {score} {run_name}\n"
                results.write(result_string)
            
            results.close()
            
    print(f"All queries retrieved. Results saved to { result_file }.")

# min-max normalization (first result will always be 1 and last result will always be 0)
def normalize_scores(response):
    scores = [hit['_score'] for hit in response['hits']['hits']]
    max_score = max(scores)
    min_score = min(scores)
    
    return [(score - min_score) / (max_score - min_score) if max_score > min_score else 0 for score in scores]



# TESTS ---------------------------------------------------------------------
# es = Elasticsearch("http://localhost:9200")

# run_name = "test_run"
# query_file = "data/queries_test.jsonl"
# result_file = "data/results_test.txt"
# index_name = "inverted_index"
# run_queries(es, run_name, query_file, result_file, index_name)