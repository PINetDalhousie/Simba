'''
This class is responsible for representing the dataset.
'''
import pandas as pd
from typing import Tuple
import joblib
from sklearn.preprocessing import MinMaxScaler

class KPIData:

    def __init__(self,path_to_txt:str) -> None:
        '''
        Initializes dataset object

        Attributes:
            data_df: a pandas dataframe to represent kpi dataset with faultcause
        '''
        self.data_df = pd.read_csv(path_to_txt,sep='\s+',comment='%',
                                   names=['Retainability', 'HOSR', 'RSRP', 'RSRQ', 'SINR', 'Throughput', 'Distance', 'FaultCause'])


    def get_df(self) -> pd.DataFrame:
        '''
        Returns current dataframe contained by this class
        '''
        return self.data_df
    

    def convert_data_for_anomaly_detection(self) -> None:
        '''
        Convert data into anomaly or normal dataset
        '''
        self.data_df['FaultCause'] = self.data_df['FaultCause'].apply(lambda x: 0.0 if x==7.0 else 1.0)


    def seperate_into_kpis_and_label(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        '''
        Seperates current dataframe into kpis and labels so that a NN can be trained
        '''
        # Seperate data into kpis and label
        kpis = self.data_df.drop(["FaultCause"],axis=1)
        labels = self.data_df['FaultCause']
        # set current dataframe as none
        self.data_df = None
        return kpis, labels


    @staticmethod
    def min_max_scale_train_val(kpis_train:pd.DataFrame,kpis_val:pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
        '''
        Scales using min max scaler and saves scalar
        '''
        min_max_scaler = MinMaxScaler()
        kpis_train = min_max_scaler.fit_transform(kpis_train)
        kpis_val = min_max_scaler.transform(kpis_val)
        joblib.dump(min_max_scaler,"../scalers/min_max_scaler.save")
        return kpis_train,kpis_val
    

    @staticmethod
    def scale_test(path_to_scaler:str,test_data:pd.DataFrame) -> pd.DataFrame:
        '''
        Loads a scikit learn scaler using path_to_scaler and transforms test_data using scaler
        '''
        scaler = joblib.load(path_to_scaler)
        return scaler.transform(test_data)


if __name__ == '__main__':
    pass