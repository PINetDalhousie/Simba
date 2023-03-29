'''
This class is responsible for representing the dataset.
'''
import pandas as pd

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
    


    
if __name__ == '__main__':
    pass