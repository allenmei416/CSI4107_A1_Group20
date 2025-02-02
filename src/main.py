# Run the entire pipeline

from elasticsearch import Elasticsearch
import preprocessing as preprocess
import indexing as index
import retrieval as retrieval

run_name = "test_run"
corpus_file = "data/corpus.jsonl"
query_file = "data/queries.jsonl"
result_file = "data/results.txt"
index_name = "inverted_index"

# elasticsearch client
es = Elasticsearch("http://localhost:9200")

# 1 - preprocessing
preprocess.preprocess_corpus(corpus_file, preprocess.output_file)

# # 2 - indexing
index.create_index(es, index_name)

# 3 - retrieval
print("Running queries...")
retrieval.run_queries(es, run_name, query_file, result_file, index_name)