import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = []
  
def embed_docs(corpus_file): 
    # read corpus
    documents = []
    with open(corpus_file, 'r') as file:
        for i, line in enumerate(file):
            doc = json.loads(line)
            documents.append(doc)
        
    # get embeddings for all docs once, before queries
    for doc in documents:
        doc_content = doc['title'] + " " + doc['text']
        doc_embedding = model.encode(doc_content, convert_to_tensor=True)
        doc_embeddings.append((doc['_id'], doc_embedding))
        
    return doc_embeddings


def run_queries(run_name, corpus_file, query_file, result_file):
    # initialize document embeddings
    print('Embedding documents...')
    embed_docs(corpus_file)
    
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
            query_embedding = model.encode(query_string, convert_to_tensor=True)
        
            # calculate similarity 
            similarities = []
            for doc_id, doc_embedding in doc_embeddings:
                similarity = util.pytorch_cos_sim(query_embedding, doc_embedding)[0][0].item()
                similarities.append((doc_id, similarity))

            # sort similarity scores
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # write top 100 results to file
            with open(result_file, "a") as results:
                rank = 1
                for doc_id, similarity in similarities[:100]:
                    result_string = f"{id} Q0 {doc_id} {rank} {similarity} {run_name}\n"
                    results.write(result_string)
                    rank += 1
    
# test
# run_queries('test_bertOnly', 'data/corpus.jsonl', 'data/queries.jsonl', 'data/results/test_bertonly.txt')