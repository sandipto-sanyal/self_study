import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pkl
import gcsfs

import constants

class Train:
    def __init__(self,
                 file,
                 username
                 ):
        '''
        Reads the file sent through requests

        Parameters
        ----------
        file : werkzeug.datastructures.FileStorage or str
            The file uploaded through request.files
            or full file path if run standalone
            
        username: str
            The username of the user deploying the model

        Returns
        -------
        None.

        '''
        # read the data
        self.df = pd.read_csv(file)
        self.username = username
    
    def split_data(self):
        '''
        Splits the data into training and test dataset
        Put 25% of training data into test set

        Returns
        -------

        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.iloc[:,:-1],
                                                                                self.df.iloc[:,-1],
                                                                                test_size = 0.25
                                                                                )
    
    def start_training(self):
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)
    
    def evaluation(self):
        y_pred = self.reg.predict(self.X_test)
        self.mse = mean_squared_error(y_true=self.y_test, y_pred=y_pred)
        self.r2 = r2_score(y_true=self.y_test, y_pred=y_pred)
    
    def save_model_binary(self):
        model_path = constants.model_path+'_'+self.username
        # for local testing only
        # with open(model_path,'wb') as f:
        #     pkl.dump(self.reg,f)
        
        # for cloud storage
        fs = gcsfs.GCSFileSystem(project='cmt-cn-gcp-development')
        with fs.open(model_path, 'wb') as f:
            pkl.dump(self.reg,f)
    
    def main(self):
        '''
        Trains the data

        Returns
        -------
        MSE: float
            DESCRIPTION.
        R2: float
            DESCRIPTION.

        '''
        self.split_data()
        self.start_training()
        self.evaluation()
        self.save_model_binary()
        return self.mse, self.r2

if __name__ == '__main__':
    tr = Train(file=r'C:\Users\sandipto.sanyal\Desktop\ML app deployment demo\training_file.csv',
               username='local_testing'
               )
    mse, r2 = tr.main()
    print('MSE: {}, R2: {}'.format(mse,r2))