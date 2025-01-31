# Step 2. [10 points] Indexing: Build an inverted index, with an entry for each word in the vocabulary. 
# You can use any appropriate data structure (hash table, linked lists, Access database, etc.). 
# An example of possible index is presented below. Note: if you use an existing IR system, use its indexing mechanism.

# Input: Tokens obtained from the preprocessing module

# Output: An inverted index for fast access



from elasticsearch import Elasticsearch
import json

es = Elasticsearch("http://localhost:9200")

index_name = "inverted_index"

def index_documents(input_file):
    with open(input_file, "r") as infile:
        for i, line in enumerate(infile):
            doc = json.loads(line)
            
            if "tokens" in doc:
                elastic_doc = {
                    "tokens": doc["tokens"],
                }
                
                es.index(index=index_name, id=i + 1, document=elastic_doc)
            
            if (i + 1) % 1000 == 0:
                print(f"Indexed {i + 1} documents...")

    print("Indexing complete.")

input_file = "data/corpus_preprocessed.jsonl"
index_documents(input_file)


# Check the total number of documents in the index
response = es.count(index=index_name)
print(f"Total documents in index: {response['count']}")

# Retrieve a sample document from the index
# sample_doc = es.get(index=index_name, id=2)  # Replace `1` with a valid document ID
# print(json.dumps(sample_doc["_source"], indent=2))
