# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:45:21 2021

@author: sandipto.sanyal
"""
import pandas as pd
import sentence_iterator_full_scan
import numpy as np
import re
import multiprocessing as mp

taxonomy_file_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\taxonomy.csv'
taxonomy_df = pd.read_csv(taxonomy_file_path,
                          header=None,
                          index_col=0
                         )
trunc_taxonomy_df = taxonomy_df[~(taxonomy_df.index=='intent')]


def map_sentence_with_stw(row):
    index = row.name
    sentence = row.original_sentence
    sentences = np.array(re.split(r'\. |\? |\! |\n+',sentence.strip()), dtype='object')
    stws_found = []
    for mtw in trunc_taxonomy_df.index:
        stw_list = trunc_taxonomy_df.loc[mtw].dropna().to_list()
        # append the mtw to the stw list
        stw_list.append(mtw)
        subtaxonomy_list = np.array(stw_list, dtype='object')
        sentence_stw_pair = sentence_iterator_full_scan.get_sentence_c(sentences=sentences, 
                                                               subtaxonomy_list=subtaxonomy_list, 
                                                               mtw=mtw)
        # return only the STWs found in the sentence
        stws_found.extend(sentence_stw_pair[:,0])
    # return number of STWs found alongwith count
    print(index, ') Done::: ', sentence)
    return pd.Series([', '.join(stws_found), len(stws_found)])

training_dataset_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\training_data.xlsx'
training_df = pd.read_excel(training_dataset_path)


def perform_apply_on_df_chunks(chunk_df):
    chunk_df[['stws_found_on_full_scan','number_of_stws']] = chunk_df.apply(map_sentence_with_stw, axis=1)
    return chunk_df

n_processes = 10
chunks_of_df = np.array_split(training_df.head(100),n_processes)
p = mp.Pool(n_processes)
pool_results = p.map(perform_apply_on_df_chunks, chunks_of_df)
final_df = pd.concat(pool_results)
