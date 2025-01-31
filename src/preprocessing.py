# Step1. [5 points]  Preprocessing:  Implement preprocessing functions for tokenization and stopword removal. 
# The index terms will be all the words left after filtering out markup that is not part of the text, punctuation 
# tokens, numbers, stopwords, etc. Optionally, you can use the Porter stemmer to stem the index words. 

# Input: Documents that are read one by one from the collection

# Output: Tokens to be added to the index (vocabulary)


import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

username = os.getlogin()
nltk_path = f"C:/Users/{username}/AppData/Roaming/nltk_data"
nltk.data.path.append(nltk_path)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words("english")]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

def preprocess_corpus(input_file, output_file):
    """
    Preprocesses the entire corpus and saves the results to a new file.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for i, line in enumerate(infile):
            doc = json.loads(line)
        
            if "text" in doc:
                doc["tokens"] = preprocess(doc["text"])
                outfile.write(json.dumps(doc) + "\n")
            else:
                print(f"Warning: Document {i + 1} is missing the 'text' field.")

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} documents...")

    print(f"Preprocessing complete. Results saved to {output_file}.")


input_file = "data/corpus2.jsonl"
output_file = "data/corpus_preprocessed.jsonl"
preprocess_corpus(input_file, output_file)