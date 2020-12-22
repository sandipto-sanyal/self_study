# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:02:13 2020

@author: sandipto.sanyal
"""

import spacy
import pandas as pd
import numpy as np
nlp = spacy.load('en_core_web_md', disable=["tagger", "parser", "ner"])

def standardize_texts(text:str):
    doc = nlp(text)
    lemmatized_words = []
    for token in doc:
        if not token.is_stop:
            lemmatized_words.append(token.lemma_)
    print(text)
    return ' '.join(lemmatized_words)

def process_df_chunks(df_chunk: pd.DataFrame,
                      function
                     ):
    df_chunk['lemmatized_sentences'] = df_chunk.Plot.apply(function)
    return df_chunk

import multiprocessing
n_processes = multiprocessing.cpu_count()-2
df = pd.read_csv('./datasets/wiki_movie_plots_deduped.csv', encoding='utf-8')
df_split = np.array_split(df.head(200), n_processes)
pool = multiprocessing.Pool(n_processes)
df_lemmatized = pd.concat(pool.map(process_df_chunks, df_split))
pool.close()
pool.join()
df_lemmatized.to_csv('./datasets/wiki_movie_plots_deduped_lemmatized.csv', index=False)