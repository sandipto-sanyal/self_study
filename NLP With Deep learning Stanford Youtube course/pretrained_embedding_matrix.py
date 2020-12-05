# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:54:10 2020

@author: sandipto.sanyal
"""
import numpy as np
from typing import Tuple, Dict

def load_glove_vectors(glove_path:str,
                       word_index: dict
                       ) -> Tuple[Dict, np.ndarray, int]:
    '''
    

    Parameters
    ----------
    glove_path : str
        The path of glove vector file.
    word_index : dict
        The word index of words in vocabulary.
        Format: Ideally Keras Tokenizer generates word indices by ranking them
        in descending order of counts in corpus
        {'word1':rank_word1,
         'word2':rank_word2
         }

    Returns
    -------
    embeddings_index : Dict
        Dictionary in the form {'word1':np.ndarray(embedding_vector1)}.
    embedding_matrix : np.ndarray
        Numpy array with shape (v+1,e)
        where v = number of words in vocabulary
        e = embedding dimension
    embedding_dim : int
        The embedding dimension of the word vectors

    '''
    # to return embeddings_index, embedding_matrix
    embeddings_index = {}
    f = open(glove_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_dim = list(embeddings_index.values())[0].shape[0]
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    print('Info: Dimensionality of word vector: {}'.format(embedding_dim))
    return embeddings_index, embedding_matrix, embedding_dim

if __name__ == '__main__':
    glove_dir = r'C:\Users\sandipto.sanyal\OneDrive - Accenture\Documents\Study materials\self study\text_summarization\glove.6B.50d.txt'
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    corpus = ['This is USD.']
    tok = Tokenizer()
    tok.fit_on_texts(corpus)
    X = tok.texts_to_sequences(corpus)
    X = pad_sequences(X, maxlen=5, padding='pre',truncating='pre')
    test_word_index = tok.word_index
    embeddings_index, embedding_matrix, embedding_dim = load_glove_vectors(glove_dir, test_word_index)
    print(embeddings_index['bank'])