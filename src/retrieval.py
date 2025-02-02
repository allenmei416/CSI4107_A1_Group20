# Step 3. [10 points] Retrieval and Ranking:  Use the inverted index (from step 2) to find the 
# limited set of documents that contain at least one of the query words. Compute the cosine 
# similarity scores between a query and each document. 

# Input: One query and the Inverted Index (from Step2)

# Output: Similarity values between the query and each of the documents. Rank the documents in decreasing order of similarity scores.

from elasticsearch import Elasticsearch
import json

es = Elasticsearch("http://localhost:9200")

run_name = "test_run"
input_file = "data/queries.jsonl"
output_file = "data/results.txt"
index_name = "inverted_index"

def run_queries():
    # create output file
    output = open(output_file, "w")
    output.close()

    # read queries from file
    with open(input_file, "r") as infile:
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
            
            # write results to file
            results = open(output_file, "a")
            rank = 0

            for doc in response["hits"]["hits"]:
                doc_id = doc['_id']
                score = doc["_score"] # TODO: normalize score so that its between 0 and 1
                rank += 1

                result_string = id + "\tQ0\t " + doc_id + "\t" + str(rank) + "\t" + str(score) + "\t" + run_name + "\n"
                results.write(result_string)
            
            results.close()



# TESTS ---------------------------------------------------------------------
run_queries()