# Run the entire pipeline

from elasticsearch import Elasticsearch
#import preprocessing as preprocess
import indexing as index
import retrieval as retrieval
import rerank_BERT as rerank_BERT
import rerank_cross_encoder as rerank_cross_encoder
import rerank_BERT_BM25 as rerank_BERT_BM25

run_name = "test_run"
corpus_file = "../data/corpus.jsonl"
query_file = "../data/queries.jsonl"
result_file = "../data/results/results.txt"
result_file_bert = "../data/results/results_bert.txt"
result_file_bert_BM25 = "../data/results/results_bert_BM25.txt"
result_file_cross_encoder = "../data/results/results_cross_encoder.txt"
output_file = "../data/corpus_preprocessed.jsonl"
index_name = "inverted_index"

# elasticsearch client
es = Elasticsearch("http://localhost:9200")

# 1 - preprocessing
#preprocess.preprocess_corpus(corpus_file, output_file)

# # 2 - indexing
#index.create_index(es, corpus_file, index_name)

# 3 - retrieval
# print("Running queries...")
# retrieval.run_queries(es, run_name, query_file, result_file, index_name)

# 4 - reranking with BERT SentenceTransformer
# print("Running queries...")
# rerank_BERT.run_queries_BERT(es, run_name, query_file, result_file_bert, index_name)


# 5 - reranking with Cross Encoder
# print("Running queries...")
# rerank_cross_encoder.run_queries_cross_encoder(es, run_name, query_file, result_file_cross_encoder, index_name)

# 6 - reranking with BERT SentenceTransformer and BM25
print("Running queries...")
rerank_BERT_BM25.run_queries_BERT(es, run_name, query_file, result_file_bert_BM25, index_name)