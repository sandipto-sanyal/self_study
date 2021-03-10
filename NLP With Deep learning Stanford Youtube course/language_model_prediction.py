# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:14:29 2021

@author: sandipto.sanyal
"""
import pickle as pkl
import keras
import re
from tqdm import tqdm

from text_cleaners import LanguageModelTextCleaner
class LanguageModelPrediction:
    def __init__(self,
                 seed_text:str,
                 model_folder: str
                 ):
        self.model_folder = model_folder
        self.seed_text = seed_text
        self.load_binaries()
        
    def load_binaries(self):
        with open('./binary_files/{}/label_encoder.pkl'.format(self.model_folder),'rb') as f:
            self.le = pkl.load(f)
        with open('./binary_files/{}/tokenizer.pkl'.format(self.model_folder),'rb') as f:
            self.tok = pkl.load(f)
        
        self.model = keras.models.load_model('./binary_files/{}/language_model.model'.format(self.model_folder))
    
    def clean_seed_text(self):
        lmtc = LanguageModelTextCleaner(text=self.seed_text)
        self.seed_text = lmtc.cleaner()
        
    def tokenize_pad(self):
        # get the shape of the input layer
        sequence_length = self.model.layers[0].batch_input_shape[1]
        # if seed text length is > allowable sequence length on which model was trained then truncate the sequence
        self.X_test = keras.preprocessing.sequence.pad_sequences(self.tok.texts_to_sequences([self.seed_text]),
                                                                 maxlen=sequence_length,
                                                                 )
        
    def perform_prediction_of_next_word(self):
        y_pred = self.model.predict(self.X_test)
        y_classes = y_pred.argmax(axis=-1)
        next_word = self.le.label_encoder.inverse_transform(y_classes)
        return next_word  
    
    def inverse_string_cleaner(self, 
                               string:str):
        string = re.sub(' eosquotes','\'',string)
        string = re.sub(' eosstop','. ',string)
        string = re.sub(' eosexcl','! ',string)
        string = re.sub(' eosques','? ',string)
        string = re.sub(' eoscomma',', ',string)
        string = re.sub(' +',' ', string)
        return string
    
    def generate_text(self, 
                      n_words:int=10):
        self.clean_seed_text()
        for iteration in tqdm(range(0,n_words)):
            self.tokenize_pad()
            next_word = self.perform_prediction_of_next_word()[0]
            self.seed_text = self.seed_text + " " + next_word
        self.seed_text = self.inverse_string_cleaner(string=self.seed_text)
        return self.seed_text
    

if __name__ == '__main__':
    seed_text='North Carolina'
    lmp = LanguageModelPrediction(seed_text=seed_text, model_folder='trump speech')
    seed_text = lmp.generate_text(n_words=1000)
    print('\n',seed_text)
    
        
