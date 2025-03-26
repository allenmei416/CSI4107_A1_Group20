import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

def embed_docs(corpus_file):
    # read corpus
    documents = []
    with open(corpus_file, 'r') as file:
        for i, line in enumerate(file):
            doc = json.loads(line)
            documents.append(doc)
    
    # tokenize documents (BM25)
    tokenized_docs = []
    for doc in documents:
        doc_content = doc['title'] + " " + doc['text']
        tokenized_doc = word_tokenize(doc_content.lower())
        tokenized_docs.append(tokenized_doc)

    bm25 = BM25Okapi(tokenized_docs)

    return bm25, documents


def run_queries(run_name, corpus_file, query_file, result_file):
    # initialize document embeddings and BM25
    print('Embedding documents with BM25...')
    bm25, documents = embed_docs(corpus_file)
    
    # create output file
    output = open(result_file, "w")
    output.close()
    
    # run queries
    print('Running queries...')
    with open(query_file, "r") as infile:
        for i, line in enumerate(infile): 
            query = json.loads(line)
            id = query['_id']
            query_string = query['text']
            query_tokens = word_tokenize(query_string.lower()) 
            
            # calculate BM25 scores
            scores = bm25.get_scores(query_tokens)
            similarities = [(documents[idx]['_id'], score) for idx, score in enumerate(scores)]

            # sort similarity scores
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # write top 100 results to file
            with open(result_file, "a") as results:
                rank = 1
                for doc_id, score in similarities[:100]:
                    result_string = f"{id} Q0 {doc_id} {rank} {score} {run_name}\n"
                    results.write(result_string)
                    rank += 1

# test
# run_queries('test_bm25', 'data/corpus.jsonl', 'data/queries.jsonl', 'data/results/test_bm25.txt')
