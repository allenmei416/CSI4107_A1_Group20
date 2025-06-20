# Step 2. [10 points] Indexing: Build an inverted index, with an entry for each word in the vocabulary. 
# You can use any appropriate data structure (hash table, linked lists, Access database, etc.). 
# An example of possible index is presented below. Note: if you use an existing IR system, use its indexing mechanism.

# Input: Tokens obtained from the preprocessing module

# Output: An inverted index for fast access



from elasticsearch import Elasticsearch
import json
import time

body = {
	"settings": {
	  "number_of_shards": 1,

	  "analysis": {
	    "analyzer": {
        "my_analyzer": {
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "stop",
            "porter_stem"
          ],
          "char_filter": ["html_strip", "number_filter"]
        }	  
      },
      "char_filter":{
          "number_filter":{
            "type":"pattern_replace",
            "pattern":"\\d+",
            "replacement":""
        }
      }, 
      "similarity": {
        "custom_bm25": { 
          "type": "BM25",
          "b":    0 , # b and k1 can be changed to tune indexing; index must be rebuilt after
          "k1" : 1.2 
        }
      }
	  }
	},
    "mappings": {
      "properties": {
        "id": {
          "type": "keyword"
        },
        "title": {
          "type": "text"
        },
        "text": {
          "type": "text"
        }
	  }
	}
}

def index_documents(es, input_file, index_name):
    with open(input_file, "r") as infile:
        for i, line in enumerate(infile):
            doc = json.loads(line)
            ids = doc.pop('_id')
            doc['id'] = ids
            es.index(index=index_name, document=doc, id=ids)
            
            if (i + 1) % 1000 == 0:
                print(f"Indexed {i + 1} documents...")

    print("Indexing complete.")


def create_index(es, input_file, index_name):
    # init index, delete if it already exists then create
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=body)

    # add documents to index
    index_documents(es, input_file, index_name)



# TESTS ---------------------------------------------------------------------
# es = Elasticsearch("http://localhost:9200")

# input_file = "../data/corpus.jsonl"

# index_name = "inverted_index"
# create_index(es, input_file, index_name)

# # wait for index to complete
# time.sleep(2)

# # Check the total number of documents in the index
# response = es.count(index=index_name)
# print(f"Total documents in index: {response['count']}")

# # Retrieve a sample document from the index
# sample_doc = es.get(index=index_name, id=0)  # Replace `1` with a valid document ID
# print(json.dumps(sample_doc["_source"], indent=2))
