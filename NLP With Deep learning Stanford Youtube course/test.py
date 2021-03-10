# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:22:40 2021

@author: sandipto.sanyal
"""
import json
t = lm.tok.word_index
s = json.dumps(tok.word_index)
with open('vocab.txt','w') as f:
    f.write(s)


#=================================================
from label_encoder_ext import LabelEncoderExt
country_list = ['Argentina', 'Australia', 'Canada', 'France', 'Italy', 'Spain', 'US', 'Canada', 'Argentina, ''US']

label_encoder = LabelEncoderExt()

label_encoder.fit(country_list)
print(label_encoder.classes_) # you can see new class called Unknown
print(label_encoder.transform(country_list))


new_country_list = ['Canada', 'France', 'Italy', 'Spain', 'US', 'India', 'Pakistan', 'South Africa']
print(label_encoder.transform(new_country_list))