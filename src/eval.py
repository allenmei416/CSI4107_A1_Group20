import pytrec_eval
import json
from collections import defaultdict

qrel_file = 'data/qrels/test.tsv'
run_file = 'data/results.txt'
output_file = 'data/eval.'

# modified pytrec_eval.parse_qrel to better fit our file format
def parse_qrel(f_qrel):
    qrel = defaultdict(dict)
    f_qrel.readline()

    for line in f_qrel:
        query_id, object_id, relevance = line.strip().split()

        assert object_id not in qrel[query_id]
        qrel[query_id][object_id] = int(relevance)

    return qrel


with open(qrel_file, 'r') as qrel_f:
    qrel = parse_qrel(qrel_f)

with open(run_file, 'r') as run_f:
    run = pytrec_eval.parse_run(run_f)

evaluator = pytrec_eval.RelevanceEvaluator(
    qrel, {'map'})

# MAP score of all test queries
eval_results = evaluator.evaluate(run)

# get average
map_values = [query['map'] for query in eval_results.values()]
average_map = sum(map_values) / len(map_values)
print(average_map)

'''
TODO: run only titles and both titles/full text
calculate MAP for both and compare

optional: play around with the BM25 parameters and see if they give better results overall
'''
