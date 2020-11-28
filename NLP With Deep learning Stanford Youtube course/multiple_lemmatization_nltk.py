# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:25:57 2020

@author: sandipto.sanyal
"""
import re

import nltk
nltk.data.path.append('./nltk_data')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

semantically_important_sw = ['not']
    
class StringCleaner:
    def __init__(self):
        '''Cleans the given string.
        The following cleaning of string are done
        
        Returns
        -------
        None.

        '''        
        # self.nlp = spacy.load("en_core_web_sm")
        self.list_of_stopwords = set((stopwords.words('english')))
        
    def lemmatize_all(self, sentence):
        wnl = WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag.startswith("NN"):
                yield wnl.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                yield wnl.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                yield wnl.lemmatize(word, pos='a')
            else:
                yield word
    
    def cleaner(self,sentence:str) -> tuple:
        '''Takes in a raw sentence and cleans it. The following cleaner are
        applied:
            1. replace special characters in the sentence and \n by space
            2. strip white spaces
            3. Remove extra spaces in sentence
            4. Lower case the string
            5. Tokenize
            6. Lemmatized tokens
        

        Parameters
        ----------
        sentence : str
            Unclean sentence.

        Returns
        -------
        tuple
            tuple containing list of lemmatized tokens, and the modified string

        '''
        sentence = self.cleaner_v2(sentence)
        lemmatized_sentence = self.lemmatize_all(sentence)
        tokenized_list_lemmatized = [word for word in lemmatized_sentence \
                                     if not (word in self.list_of_stopwords \
                                             and word not in semantically_important_sw)]
        # lemmatized sentence
        sentence_lemmatized = ' '.join(tokenized_list_lemmatized)
        return tokenized_list_lemmatized, sentence_lemmatized

    
    def clean_keeping_stopwords(self, sentence: str) -> str:
        '''
        Text cleaner without stop words removal
        1. Decontraction and lowering
        2. Lemmatization

        Parameters
        ----------
        sentence : str
            Uncleaned text

        Returns
        -------
        str
            Cleaned lemmatized string

        '''
        sentence = self.cleaner_v2(sentence)
        lemmatized_sentence = self.lemmatize_all(sentence)
        tokenized_list_lemmatized = lemmatized_sentence
        
        cleaned_sentence = ' '.join(tokenized_list_lemmatized)
        return cleaned_sentence
        
    def cleaner_v2(self,sentence:str) -> str:
        '''
        Takes in a raw sentence and cleans it.

        Parameters
        ----------
        sentence : str
            Unclean sentence.

        Returns
        -------
        tuple
            tuple containing list of lemmatized tokens, and the modified string

        '''
        def custom_replacements(text: str) -> str:
            '''
            This method defines some of the common abbreviations like U.S., U.N. to be expanded
        
            Parameters
            ----------
            text : str
                Text having words like U.S. U.N. etc.
        
            Returns
            -------
            str
                DESCRIPTION.
        
            '''
            # country name replacement
            text = re.sub('U\.S\.A\.|U\.S\.A|USA', 'United States of America', text)
            text = re.sub('U\.S\.|U\.S|US', 'United States', text)
            text = re.sub('U\.N\.|U\.N|UN', 'United Nations',text)
            text = re.sub('U\.A\.E\.|U\.A\.E|UAE', 'United Arab Emirates', text)
            text = re.sub('U\.K\.|U\.K|UK', 'United Kingdom', text)
            
            text = re.sub('\$','dollar ', text)
            # replace numbers and digits with ' '
            text = re.sub('\d+\.\d+|\d+',' number ', text)
            # replace '-'
            text = re.sub('-',' ', text)
            return text
        
        def deep_text_cleaner(text: str) -> str:
            '''
            Cleans the text by
            1. Changing the end of sentence tokens to add space between them and the words.
            2. Other special characters to be removed
        
            Parameters
            ----------
            text : str
                Unclean text
        
            Returns
            -------
            str
                Cleaned text
        
            '''
            # some custom replacements
            text = custom_replacements(text)
            # lower case
            text = text.lower()
            # expand contractions
            def decontracted(phrase):
                # specific
                phrase = re.sub(r"won\'t", "will not", phrase)
                phrase = re.sub(r"can\'t", "can not", phrase)
        
                # general
                phrase = re.sub(r"n\'t", " not", phrase)
                phrase = re.sub(r"\'re", " are", phrase)
                phrase = re.sub(r"\'s", " is", phrase)
                phrase = re.sub(r"\'d", " would", phrase)
                phrase = re.sub(r"\'ll", " will", phrase)
                phrase = re.sub(r"\'t", " not", phrase)
                phrase = re.sub(r"\'ve", " have", phrase)
                phrase = re.sub(r"\'m", " am", phrase)
                return phrase
            text = decontracted(text)
            
            # create a transformation dictionary to put a space before the special characters encountered

            text = re.sub('[^a-zA-Z0-9]',' ', text)
            
            # replace multiple spaces with a single space
            text = re.sub(' +',' ', text)
            # strip the sentence
            text = text.strip()
            return text
        
        return deep_text_cleaner(sentence)
        
if __name__ == '__main__':
    sc = StringCleaner()
    component_text = 'he is not battling with cancer'
    print(sc.cleaner(component_text))
