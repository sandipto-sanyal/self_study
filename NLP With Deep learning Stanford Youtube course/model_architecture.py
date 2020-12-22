# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:49:50 2020

@author: sandipto.sanyal
"""
import keras
import keras.backend as K

import constants
import pretrained_embedding_matrix

def create_model(word_index: dict):
        '''
    Create a new keras model

    Parameters
    ----------
    word_index : dict
        The word index created by keras tokenizer.

    Returns
    -------
    keras model
        DESCRIPTION.

    '''
        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            get_precision = true_positives / (predicted_positives + K.epsilon())
            return get_precision
        
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            get_recall = true_positives / (possible_positives + K.epsilon())
            return get_recall
            
        def f1(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
            return f1_val
        
        #
        
        inputs = keras.layers.Input(shape=(constants.max_length,), dtype="int32")
    
        # embed with Glove 50D embedding layer
        embeddings_index, embedding_matrix, embedding_dim = pretrained_embedding_matrix.load_glove_vectors(constants.GLOVE_DIR,word_index)
        vocab_size = len(word_index) + 1

        x = keras.layers.Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=constants.max_length,
                            trainable=False
                            )(inputs)

        # Add 2 bidirectional LSTMs
        x = keras.layers.SimpleRNN(64, return_sequences=True)(x)
        x = keras.layers.Dropout(0.30)(x)
        x = keras.layers.SimpleRNN(32)(x)
        x = keras.layers.Dropout(0.30)(x)
        
        # convolution layers
        # x = keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu')(x)
        # x = keras.layers.MaxPooling1D(2)(x)
        # x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(50, activation='relu')(x)
        # x = keras.layers.Dropout(0.35)(x)
        # Add a classifier
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        
        
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy',
                                                                   precision,
                                                                   recall,
                                                                   f1
                                                                   ])
        
        print('\nModel summary:::\n')
        print(model.summary())
        return model