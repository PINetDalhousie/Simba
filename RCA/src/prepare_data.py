import pandas as pd
#import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pandas as pd

class PrepareData:

    def __init__(self) -> None:
        # dict to store dataframes
        self.faults = {}
        pass

    def read_data(self, path):
        '''
        Read data from csv files
        '''
        # get list of csv files in path    
        faults = [f for f in os.listdir(path) if f.endswith('.csv')]

        # iterate over csv files
        for fault in faults:
            # split fault name
            fault_name = fault.split('-')[0]
            # read csv
            self.faults[fault_name] = pd.read_csv(path+fault)

    def cast_cell_to_int(self):
        # cast cell id to int
        for key, value in self.faults.items():
            value['servingCell:vector'] = np.floor(value['servingCell:vector']).astype(int)
        
    def cast_timestamp_to_int(self):
        # multiply by 10^len(decimal places)
        for key, value in self.faults.items():
            value['timestamp:vector'] = value['timestamp:vector'].apply(lambda x: int(x * 10**len(str(x).split('.')[-1])))

    def aggregate(self):
        # aggregate using cell id and timestamp
        for key, value in self.faults.items():
            value = value.groupby(['timestamp:vector','servingCell:vector']).mean().reset_index()


    def label_normal(self):
        '''
        Label normal data
        '''
        self.faults['normal']['label'] = 0

    def label_epr(self,cell_id):
        '''
        Label epr data
        '''
        self.faults['epr']['label'] = 0
        self.faults['epr']['label'] = self.faults['epr'].apply(lambda x: 1 if x['servingCell:vector']==cell_id else x['label'],axis=1)

    def label_interference(self,cell_id):
        '''
        Label interference data
        '''
        self.faults['interference']['label'] = 0
        self.faults['interference']['label'] = self.faults['interference'].apply(lambda x: 2 if x['servingCell:vector']==cell_id else x['label'],axis=1)


    def merge(self):
        '''
        Merge all dataframes
        '''
        # combine all dataframes in the dictionary into one
        combined_df = pd.concat(self.faults.values(), ignore_index=True)
        return combined_df

    @staticmethod
    def remove_columns(df):
        '''
        Remove timestamp, cell id, and ue id
        '''
        df.drop(
            [
                'timestamp:vector',
                'servingCell:vector',
                'UEid:vector',
                'Unnamed: 0'
            ],
            axis=1,
            inplace=True
            )
        return df


def prepare_data():
    prepare_data = PrepareData()

    prepare_data.read_data(
        '../data/June-28/'
        )
    
    # cast columns to int
    prepare_data.cast_cell_to_int()
    prepare_data.cast_timestamp_to_int()

    # aggregate using cell id and timestamp
    prepare_data.aggregate()

    # label data
    prepare_data.label_normal()
    prepare_data.label_epr(cell_id=2)
    prepare_data.label_interference(cell_id=5)

    # combine dataframes
    combined_df = prepare_data.merge()

    # remove timestamp, cell id, and ue id
    combined_df = PrepareData.remove_columns(combined_df)
    
    # save to csv
    combined_df.to_csv('../data/combined.csv', index=False)
    

def prepare_solar_data():
    solar = pd.read_csv('../MTGNN/data/solar_AL.txt', sep=',')
    print(solar.head())

def main():

    # Read data
    normal_df = pd.read_csv('../data/June-12/normal-v1.csv')
    epr_df = pd.read_csv('../data/June-12/epr-v1.csv')
    #interference_df = pd.read_csv('../data/April-25/clean-data-simu5g-Interference-downtilt.csv')
    #tlho_df = pd.read_csv('../data/April-25/clean-data-simu5g-TLHO.csv')
    
    # Add label column
    normal_df['label'] = 0
    epr_df['label'] = 0
    #epr_df['label'] = 1
    #interference_df['label'] = 2
    #tlho_df['label'] = 3

    epr_df['servingCell:vector'] = np.floor(epr_df['servingCell:vector']).astype(int)
    epr_df['label'] = epr_df.apply(lambda x: 1 if x['servingCell:vector']==4 else x['label'],axis=1)
    epr_df['timestamp:vector'] = epr_df['timestamp:vector'].apply(lambda x: int(x * 10**len(str(x).split('.')[-1])))
    epr_df = epr_df.groupby(['timestamp:vector','servingCell:vector','label']).mean().reset_index()


    normal_df['servingCell:vector'] = np.floor(normal_df['servingCell:vector']).astype(int)
    normal_df['timestamp:vector'] = normal_df['timestamp:vector'].apply(lambda x: int(x * 10**len(str(x).split('.')[-1])))
    normal_df = normal_df.groupby(['timestamp:vector','servingCell:vector','label']).mean().reset_index()

    # Combine tables
    df = pd.concat([normal_df,epr_df],axis=0)

    # save to csv
    df.to_csv('../data/June-12/combined.csv', index=False)
    print(asd)
    # Remove NaN due to throughput
    df = normal_df.dropna()

    # Create plots

    # Preprocess data
    df['servingCell:vector'] = df['servingCell:vector'].astype(int)

    # Simulation time is in decimal and can vary in the number of decimal places
    # Convert to integer by multiplying by 10^number of decimal places
    # e.g., 0.1 -> 1, 0.01 -> 10, 0.001 -> 100
    df['timestamp:vector'] = df['timestamp:vector'].apply(lambda x: int(x * 10**len(str(x).split('.')[-1])))

    # Aggregate UE data to base stations
    df = df.groupby(['timestamp:vector','servingCell:vector','label']).mean().reset_index()
    
    #df = df[df['label']==3][df['servingCell:vector']==6]
    
    # print(df.head())
    # print(df['servingCell:vector'].unique())
    # print(df.info())
    
    # Plot 'feature' column against 'time' column
    df.plot(x='timestamp:vector', y='servingRSRP:vector', marker='o')

    # Set plot title and axis labels
    plt.title('Feature Variation Over Time')
    plt.xlabel('Time')
    plt.ylabel('Feature')

    # Save the plot as an image file (e.g., PNG)
    plt.savefig('../data/plots/plot.png')


    df.to_csv('../data/TLHO.csv', index=False)

    sys.exit()
    # Filter out the groups that do not have all base stations
    base_stations = df['servingCell:vector'].unique()
    filtered_groups = []
    for _, group in grouped:
        if set(group['servingCell:vector'].unique()) == set(base_stations):
            filtered_groups.append(group)

    # Concatenate the filtered groups back into a single dataframe
    df = pd.concat(filtered_groups)
    # Reset the index of the filtered dataframe
    df.reset_index(drop=True, inplace=True)
    all_df.append(df)

    df = pd.concat(all_df)

    print(df.head())
    df.to_csv('../data/data.csv', index=False)


    

    # Aggregate temporal data so that we have per base station per unit time
    # print(df['servingCell:vector'].unique())
    # df = df[df['servingCell:vector']==6.]
    # df = df.sort_values(by='timestamp:vector')
    # print(df['timestamp:vector'])


    # stop the program here
    sys.exit()

    # Write data to tfrecord
    writer = tf.io.TFRecordWriter("example.tfrecords")
    for row in df.values:
        features = row.tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': tf.train.Feature(float_list=tf.train.FloatList(value=features))
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    #prepare_solar_data()
    # main()
    prepare_data()