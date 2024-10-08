import pandas as pd
import numpy as np
import re
import argparse
import os

import nbformat as nbf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from transformers import BertTokenizer, BertModel

import torch
import pickle

def main_train(technique, dir_name):
    train_data_path = "raw-dataset/train.csv"
    train_raw_df = pd.read_csv(train_data_path)
    X_train = train_raw_df['CONTENT']
    y_train = train_raw_df['CLASS']
    comment_id_train = train_raw_df['COMMENT_ID']

    test_data_path = "raw-dataset/test.csv"
    test_raw_df = pd.read_csv(test_data_path)
    X_test = test_raw_df['CONTENT']
    comment_id_test = test_raw_df['COMMENT_ID']    

    if technique == "CountVectorizer":
        # we count the occurence of each word in each comment. 
        print(f"creating {technique}...\n")
        vectorizer = CountVectorizer() # vectorizing object
        X_transformed_train = vectorizer.fit_transform(X_train).toarray()
        X_transformed_test = vectorizer.transform(X_test).toarray()


    elif technique == "TfidfVectorizer":
        # we count the occurence of each word in the entire corpus. 
        print(f"creating {technique}...\n")
        vectorizer = TfidfVectorizer() # vectorizing object
        X_transformed_train = vectorizer.fit_transform(X_train).toarray()
        X_transformed_test = vectorizer.transform(X_test).toarray()

    elif technique == "Word2Vec":
        # we extract the word embedding based on each comment
        print(f"tagging dataset...\n")
        comments = []
        for text in X_train:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words = cleaned_text.split()
            comments.append(words)
        print(f"creating {technique} model...\n")
        model = Word2Vec(comments, vector_size=100, window=5, min_count=1, workers=4)

        
        print(f"transforming dataset with {technique} model...\n")
        X_transformed_train = []
        for comment in comments:
            word_vectors = []
            for word in comment:
                if word in model.wv:
                    word_vectors.append(model.wv[word])
            if word_vectors:
                word_vectors = np.mean(word_vectors, axis=0)
            else:
                word_vectors = np.zeros(100)
            X_transformed_train.append(word_vectors)



        print(f"Tagging test dataset...\n")
        comments_test = []
        for text in X_test:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words = cleaned_text.split()
            comments_test.append(words)

        print(f"Transforming test dataset with {technique} model...\n")
        X_transformed_test = []
        for comment in comments_test:
            comment_vectors = []
            for word in comment:
                if word in model.wv:
                    comment_vectors.append(model.wv[word])
            if comment_vectors:
                comment_vectors = np.mean(comment_vectors, axis=0)
            else:
                comment_vectors = np.zeros(100)
            X_transformed_test.append(comment_vectors)


        

    elif technique == "Doc2Vec":
        # we extract the word embedding based on each comment
        print(f"tagging dataset...\n")
        comments = []
        for text in X_train:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words = cleaned_text.split()
            comments.append(words)
        print(f"creating {technique} model...\n")
        model = Word2Vec(comments, vector_size=100, window=5, min_count=1, workers=4)

        
        print(f"transforming dataset with {technique} model...\n")
        X_transformed_train = []
        for comment in comments:
            word_vectors = []
            for word in comment:
                if word in model.wv:
                    word_vectors.append(model.wv[word])
            if word_vectors:
                word_vectors = np.mean(word_vectors, axis=0)
            else:
                word_vectors = np.zeros(100)
            X_transformed_train.append(word_vectors)

        print(f"Tagging test dataset...\n")
        comments_test = []
        for text in X_test:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words = cleaned_text.split()
            comments_test.append(words)

        print(f"Transforming test dataset with {technique} model...\n")
        X_transformed_test = []
        for comment in comments_test:
            comment_vectors = []
            for word in comment:
                if word in model.wv:
                    comment_vectors.append(model.wv[word])
            if comment_vectors:
                comment_vectors = np.mean(comment_vectors, axis=0)
            else:
                comment_vectors = np.zeros(100)
            X_transformed_test.append(comment_vectors)

    elif technique == "BERT":
        # Step 1: Tokenize and encode the training data using the pre-trained BERT tokenizer
        print(f"Creating {technique} tokenizer...\n")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print(f"Loading {technique} model...\n")
        model = BertModel.from_pretrained('bert-base-uncased')

        # Step 2: Tokenize the training data and extract word embeddings
        print(f"Tokenizing training dataset...\n")
        inputs_train = tokenizer(list(X_train), return_tensors="pt", padding=True, truncation=True)

        print(f"Transforming training dataset with {technique} model...\n")
        with torch.no_grad():
            outputs_train = model(**inputs_train)

        # Step 3: Take the mean of the last hidden state for each input to get the final embeddings
        X_transformed_train = outputs_train.last_hidden_state.mean(dim=1).numpy()

        # Step 4: Repeat the same process for the test dataset
        print(f"Tokenizing test dataset...\n")
        inputs_test = tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True)

        print(f"Transforming test dataset with {technique} model...\n")
        with torch.no_grad():
            outputs_test = model(**inputs_test)

        # Step 5: Again, take the mean of the last hidden state for each input
        X_transformed_test = outputs_test.last_hidden_state.mean(dim=1).numpy()

        # Now you have X_transformed_train and X_transformed_test
        print(f"Train and test datasets have been transformed using {technique}.")

    else:
        raise ValueError("Invalid technique provided. Choose from CountVectorizer, TfidfVectorizer, Word2Vec, Doc2Vec, BERT.")
###########-------TRAIN-----#############
    print(f"Transformed data shape: {np.array(X_transformed_train).shape}")
    output_path = os.path.join(dir_name, f'train.pkl')
    output_path_csv = os.path.join(dir_name, f'train.csv')
    final_df_train = pd.DataFrame(X_transformed_train)
    final_df_train['CLASS'] = y_train
    final_df_train.set_index(comment_id_train, inplace=True)
    final_df_train.to_csv(output_path_csv, index=True) 
    print(f"Exporting data to {output_path}...\n")
    with open(output_path, 'wb') as file:
        pickle.dump(final_df_train, file)
    print(f"Data successfully exported to {output_path}.")

###########-------TEST--------#############
    print(f"Transformed data shape: {np.array(X_transformed_test).shape}")
    output_path = os.path.join(dir_name, f'test.pkl')
    output_path_csv = os.path.join(dir_name, f'test.csv')
    final_df_test = pd.DataFrame(X_transformed_test)
    final_df_test.set_index(comment_id_test, inplace=True)
    final_df_test.to_csv(output_path_csv, index=True) 
    print(f"Exporting data to {output_path}...\n")
    with open(output_path, 'wb') as file:
        pickle.dump(final_df_test, file)
    print(f"Data successfully exported to {output_path}.")


def create_notebook(dir_name):
    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Define the single code cell
    code = """
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd

with open('test.pkl', 'rb') as file:
    test_df = pickle.load(file)
with open('train.pkl', 'rb') as file:
    train_df = pickle.load(file)
    """

    # Add the cell to the notebook
    nb['cells'] = [nbf.v4.new_code_cell(code)]

    # Write the notebook to a file
    output_path = os.path.join(dir_name, f'modelling.ipynb')
    with open(output_path, 'w') as f:
        nbf.write(nb, f)

    print("Jupyter notebook 'modelling.ipynb' created successfully.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--technique",
        type=str,
        required=True,
        help="choose the technique: CountVectorizer, TfidfVectorizer, Word2Vec, Doc2Vec, BERT"
    )
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="choose an unique id for the word representation"
    )

    args = parser.parse_args()
    # Create a folder (if it doesn't already exist)
    dir_name = f'{args.technique}_{args.id}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    main_train(args.technique, dir_name)
    create_notebook(dir_name)
