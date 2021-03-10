# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:52:33 2021

@author: sandipto.sanyal
"""
import re
class LanguageModelTextCleaner:
    def __init__(self,
                 text: str
                 ):
        self.text = text
        
    def cleaner(self):
        '''
        Puts spaces in front of special characters

        Returns
        -------
        None.

        '''
        self.text = re.sub('\.',' eosstop ', self.text)
        self.text = re.sub('\!',' eosexcl ', self.text)
        self.text = re.sub('\?',' eosques ', self.text)
        self.text = re.sub('\,',' eoscomma ', self.text)
        self.text = re.sub('\'|\"',' eosquotes ', self.text)
        # replace other characters by spaces
        self.text = re.sub('[^A-Z0-9a-z]',' ',self.text)
        
        self.text = re.sub(' +',' ', self.text)
        self.text = re.sub('\n+',' ', self.text)
        
        #lower case
        self.text = self.text.lower()
        return self.text