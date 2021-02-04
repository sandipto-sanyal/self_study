# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:45:21 2021

@author: sandipto.sanyal
"""
import pandas as pd
import sentence_iterator_full_scan
import numpy as np
import re
from multiprocessing import Pool,cpu_count

def map_sentence_with_stw(sentence):
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
    print('Done::: ', sentence)
    return pd.Series([', '.join(stws_found), len(stws_found)])

def custom_apply_function(row):
    index = row.name
    sentence = row.original_sentence
    print(index, ') Done::: ', sentence)
    return pd.Series([len(sentence), 1])

def perform_apply_on_df_chunks(chunk_df):
    chunk_df[['stws_found_on_full_scan','number_of_stws']] = chunk_df.apply(custom_apply_function, axis=1)
    return chunk_df

def parallelize_dataframe(df, func, n_cores=cpu_count()-3):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df



# as per the doc: Make sure that the main module can be safely imported by a new Python interpreter without causing unintended side effects (such a starting a new process)
if __name__ == '__main__':
    taxonomy_file_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\taxonomy.csv'
    taxonomy_df = pd.read_csv(taxonomy_file_path,
                              header=None,
                              index_col=0
                             )
    trunc_taxonomy_df = taxonomy_df[~(taxonomy_df.index=='intent')]
    training_dataset_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\training_data.xlsx'
    training_df = pd.read_excel(training_dataset_path)
    print('Training df loaded ', training_df.shape)
    final_df = parallelize_dataframe(training_df.head(100),perform_apply_on_df_chunks)
    save_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\training_data_with_mapped_stw.xlsx'
    final_df.to_excel(save_path, index=False)