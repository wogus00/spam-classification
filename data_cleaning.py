import re
import pandas as pd
import nltk
import os
import argparse
from tqdm import tqdm

from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.data import find

from transformers import BertTokenizer, BertModel
import torch

# if you happen to have a CUDA supported GPU HAHAHA
# if not, I don't recommend running bert because it will take forever
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download necessary resources
try:
    find('tokenizers/punkt')
    print("'punkt' is already downloaded")
except LookupError:
    print("Downloading 'punkt' tokenizer")
    nltk.download('punkt')


try:
    find('corpora/stopwords')
    print("'stopwords' is already downloaded")
except LookupError:
    print("Downloading 'stopwords' corpus")
    nltk.download('stopwords')

try:
    find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for token in tokens:
            # Lemmatize and append
            lemmatized.append(lemmatizer.lemmatize(token))
    return " ".join(lemmatized)

def remove_stop_words(tokens):
    no_stop = []
    for token in tokens:
        if token not in stopwords.words('english'):
            no_stop.append(token)
    return no_stop

def main_tfidf():
    # Import raw dataset
    df_train = pd.read_csv('raw-dataset/train.csv')
    df_test = pd.read_csv('raw-dataset/test.csv')

    # Set tokenizers
    sentence_tokenizer = PunktSentenceTokenizer()
    tb_tokenizer = TreebankWordTokenizer()

    # Tokenize
    df_train['CONTENT'].str.lower().apply(sentence_tokenizer.tokenize)
    tokenized_comments_train = df_train['CONTENT'].str.lower().apply(tb_tokenizer.tokenize)
    df_test['CONTENT'].str.lower().apply(sentence_tokenizer.tokenize)
    tokenized_comments_test = df_test['CONTENT'].str.lower().apply(tb_tokenizer.tokenize)

    # remove stopwords
    tokenized_stop_removed_comments_train = tokenized_comments_train.apply(remove_stop_words)
    tokenized_stop_removed_comments_test = tokenized_comments_test.apply(remove_stop_words)

    # lemmatize
    lemmatized_comments_train = tokenized_stop_removed_comments_train.apply(lemmatize)
    lemmatized_comments_test = tokenized_stop_removed_comments_test.apply(lemmatize)
    df_train['CONTENT'] = lemmatized_comments_train
    df_test['CONTENT'] = lemmatized_comments_test

    # Create a folder
    dir_name = 'tfidf_preprocessed_dataset'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    output_path_test = os.path.join(dir_name, f'test.csv')
    output_path_train = os.path.join(dir_name, f'train.csv')
    
    df_test.to_csv(output_path_test) 
    df_train.to_csv(output_path_train) 

    print(f"Exporting data to {dir_name}...\n")

def get_bert_embedding(sentence):
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # tokenize
    inputs = bert_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the embeddings from BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Extract the token embedding (for classification tasks)
    embeddings = outputs.last_hidden_state[:, 0, :]  # The [CLS] token is at index 0
    return embeddings.numpy()

def main_bert():
    """
    note that df['bert_embeddings'] is in str format for some reason. 
    Before model fitting, make sure to adjust that to valid values before modelling.
    
    """
    # Import raw dataset
    df_train = pd.read_csv('raw-dataset/train.csv')
    df_test = pd.read_csv('raw-dataset/test.csv')

    df_train['bert_embeddings'] = [get_bert_embedding(sentence) for sentence in tqdm(df_train['CONTENT'], desc="Extracting BERT embeddings")]
    df_test['bert_embeddings'] = [get_bert_embedding(sentence) for sentence in tqdm(df_test['CONTENT'], desc="Extracting BERT embeddings (Test)")]

    # Create a folder
    dir_name = 'bert_embedding_extracted_dataset'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    output_path_test = os.path.join(dir_name, f'test.csv')
    output_path_train = os.path.join(dir_name, f'train.csv')
    
    df_test.to_csv(output_path_test) 
    df_train.to_csv(output_path_train) 

    print(f"Exporting data to {dir_name}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--technique",
        type=str,
        required=True,
        help="choose the technique: CountVectorizer, TfidfVectorizer, Word2Vec, Doc2Vec, BERT"
    )

    args = parser.parse_args()
    if args.technique == "tfidf":
        main_tfidf()
    elif args.technique == "bert":
        main_bert()
    # elif args.technique == "word2vec":
    #     main_word2vec(args.id)
    


