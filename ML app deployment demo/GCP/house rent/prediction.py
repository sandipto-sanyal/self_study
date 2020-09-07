# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:47:18 2020

@author: sandipto.sanyal
"""

import pandas as pd
import pickle as pkl
import gcsfs

import constants

class Prediction:
    def __init__(self,
                 prediction_file,
                 username: str
                 ):
        '''
        Caculate the predictive
        rent of the houses given in .CSV file

        Parameters
        ----------
        prediction_file : werkzeug.datastructures.FileStorage or str
            The file uploaded through request.files
            or full file path if run standalone
        
        username : str
            The username of the user deploying the model

        Returns
        -------
        None.

        '''
        self.df = pd.read_csv(prediction_file)
        self.username = username
    
    def load_model(self):
        model_path = constants.model_path+'_'+self.username
        # for local testing only
        # with open(constants.model_path+'_'+self.username,'rb') as f:
        #     self.model = pkl.load(f)
        
        # for cloud storage
        fs = gcsfs.GCSFileSystem(project='cmt-cn-gcp-development')
        with fs.open(model_path, 'rb') as f:
            self.model = pkl.load(f)
        
        
    def predict(self):
        self.df['prediction'] = self.model.predict(self.df)
    
    def main(self):
        '''
        Do the prediction of uploaded CSV

        Returns
        -------
        dict
            The prediction results in JSON format.

        '''
        self.load_model()
        self.predict()
        return self.df.to_json(orient='records')

if __name__ == '__main__':
    pr = Prediction(prediction_file=r'C:\Users\sandipto.sanyal\Desktop\ML app deployment demo\prediction_file.csv',
                    username='local_testing'
                    )
    response = pr.main()