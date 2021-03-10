# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:07:35 2021

@author: sandipto.sanyal
"""
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from label_encoder_ext import LabelEncoderExt
import tensorflow as tf
import datetime
from keras.callbacks import TensorBoard
import pickle as pkl

from text_cleaners import LanguageModelTextCleaner


class LanguageModel:
    def __init__(self,
                 path: str,
                 window_size: int = 5,
                 
                 ):
        self.window_size = window_size
        self.house_keeping(path)       
    
    def house_keeping(self, 
                      path:str
                      ):
        '''
        Initial stuffs like open the text file read it and create
        necessary folders to store the model binaries

        Parameters
        ----------
        path : str
            The full path to the text file.

        Returns
        -------
        None.

        '''
        self.folder_name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r', encoding='utf-8') as file:
            self.training_text = file.read()
        # create the folder to host model
        try:
            os.mkdir('./binary_files/{}'.format(self.folder_name))
            print('Info:::: ', self.folder_name, ' created successfully')
        except Exception as e:
            print('Folder creation exception:: ', e)
            pass
        
        
    def text_preprocessing(self):
        '''
        Puts spaces in front of special characters

        Returns
        -------
        None.

        '''
        lmtc = LanguageModelTextCleaner(text=self.training_text)
        self.training_text = lmtc.cleaner()


    def create_sliding_window(self):
        
        text_array = np.array(self.training_text.split(' '))
        # initialize the slider_start position
        slider_start = 0
        slider_end = slider_start+self.window_size
        
        # initialize the df columns
        df_cols = {'context': [],
                   'next_word': []
                   }
        # loop through the sliding window
        pbar = tqdm(total = 1e4+1)
        while slider_end < len(text_array):
            # initialize the slider end
            context = ' '.join(text_array[slider_start:slider_end])
            next_word = str(text_array[slider_end])
            # increment the slider
            slider_start +=1
            slider_end = slider_start+self.window_size
            # append to the df cols
            df_cols['context'].append(context)
            df_cols['next_word'].append(next_word)
            pbar.update(1)
        
        pbar.close()
        self.df = pd.DataFrame(df_cols)
        # filter out data which has null
        self.df = self.df[~self.df.context.isna()]
            
    
    def train_test_splitting(self):
        self.train_df, self.test_df = train_test_split(self.df,test_size=0.001)
    
        
    def tokenize_pad_labelencode(self):
        '''
        Creates the keras tokenizer on the given text

        Returns
        -------
        None.

        '''
        self.tok = keras.preprocessing.text.Tokenizer(lower=False)
        self.tok.fit_on_texts(self.train_df.context)
        
        # create sequences for the training dataset
        self.X_train = keras.preprocessing.sequence.pad_sequences(self.tok.texts_to_sequences(self.train_df.context),
                                                                  maxlen=self.window_size)
        self.le = LabelEncoderExt()
        self.le.fit(self.train_df.next_word)
        self.y_train = self.le.transform(self.train_df.next_word)
        
        # for the test dataset
        self.X_test = keras.preprocessing.sequence.pad_sequences(self.tok.texts_to_sequences(self.test_df.context),
                                                                  maxlen=self.window_size)
        self.y_test = self.le.transform(self.test_df.next_word)
        
        # export the binaries
        with open('./binary_files/{}/label_encoder.pkl'.format(self.folder_name),'wb') as f:
            pkl.dump(self.le, f)
        with open('./binary_files/{}/tokenizer.pkl'.format(self.folder_name),'wb') as f:
            pkl.dump(self.tok, f)
            
             
    def model_architecture(self):
        inputs = keras.Input(shape=(self.window_size,))
        embedding = keras.layers.Embedding(len(self.tok.word_index)+1,10)(inputs)
        layer1 = keras.layers.LSTM(50, return_sequences=False)(embedding)
        # layer1 = keras.layers.Dense(10, activation='relu')(inputs)
        # add a classifier
        outputs = keras.layers.Dense(len(self.le.classes_), activation='sigmoid')(layer1)
        self.model = keras.Model(inputs,outputs)
        self.model.summary()
        
    
        
    def train_model(self):
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # es = tf.keras.callbacks.EarlyStopping(patience=10)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])
        self.model.fit(self.X_train,
                       self.y_train,
                       batch_size=32,
                       epochs=300,
                       # validation_data=(self.X_test, self.y_test),
                       callbacks=[tensorboard_callback, 
                                  # es
                                  ],
                       verbose=True
                       )
        self.model.save('./binary_files/{}/language_model.model'.format(self.folder_name))
        

if __name__ == '__main__':
    path = './datasets/trump speech.txt'
    lm = LanguageModel(path=path, window_size=7)
    lm.text_preprocessing()
    lm.create_sliding_window()
    lm.train_test_splitting()
    lm.tokenize_pad_labelencode()
    lm.model_architecture()
    lm.model.summary()
    lm.train_model()
    #tensorboard activation command: tensorboard --logdir logs/fit
    