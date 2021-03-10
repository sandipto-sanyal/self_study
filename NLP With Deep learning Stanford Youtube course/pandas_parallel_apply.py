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
import feature_extractor


def custom_apply_function(row):
    index = row.name
    sentence = row.original_sentence
    print(index, ') Done::: ', sentence)
    return pd.Series([len(sentence), 1])

def perform_apply_on_df_chunks(chunk_df):
    taxonomy_file_path = r's3://valueinsightsplatform.textanalytics.data.dev/text-analytics-taxonomy/taxonomy.csv'
    # read the taxonomy file
    taxonomy_df = pd.read_csv(taxonomy_file_path,
                              header=None,
                              index_col=0
                             )
    chunk_df[['priority_stws_full_scan',
            'n_priority_stws',
            'intent_stws_full_scan',
            'n_intent_stws',
            'numeric_encoding'
            ]]  = chunk_df.original_sentence.apply(lambda row: feature_extractor.map_sentence_with_stw(row,taxonomy_df,verbose=True))
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
    training_dataset_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\vodafone_call_feature extractor\others\training_data.xlsx'
    training_df = pd.read_excel(training_dataset_path)
    print('Training df loaded ', training_df.shape)
    # final_df = parallelize_dataframe(training_df,perform_apply_on_df_chunks)
    
    # serial apply
    final_df = perform_apply_on_df_chunks(training_df)
    save_path = r'C:\Users\sandipto.sanyal\Documents\AVS PS\TSA VIP\phase2\data\Intent and Guidance captures\CRs\vodafone_call_feature extractor\others\training_data_with_mapped_stw.xlsx'
    final_df.to_excel(save_path, index=False)
