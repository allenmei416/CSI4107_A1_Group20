# Run the entire pipeline

from elasticsearch import Elasticsearch
import preprocessing as preprocess
import indexing as index
import retrieval as retrieval

run_name = "test_run"
corpus_file = "../data/corpus.jsonl"
query_file = "../data/queries.jsonl"
result_file = "../data/results/results.txt"
output_file = "../data/corpus_preprocessed.jsonl"
index_name = "inverted_index"

# elasticsearch client
es = Elasticsearch("http://localhost:9200")

# 1 - preprocessing
preprocess.preprocess_corpus(corpus_file, output_file)

# # 2 - indexing
index.create_index(es, corpus_file, index_name)

# 3 - retrieval
print("Running queries...")
retrieval.run_queries(es, run_name, query_file, result_file, index_name)