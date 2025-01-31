from elasticsearch import Elasticsearch

# Connect to Elasticsearch using HTTP
es = Elasticsearch("http://localhost:9200")

# Check if Elasticsearch is running
if es.ping():
    print("Connected to Elasticsearch!")
else:
    print("Could not connect to Elasticsearch.")