from prepare_data import PrepareData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)



def set_time_index(df):
    """
    This function takes a DataFrame and performs two main operations:
    1. Drops the first column of the DataFrame.
    2. Converts the 'timestamp:vector' column to a timedelta, sets it as the index of the DataFrame, 
       and keeps the column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df (pd.DataFrame): The modified DataFrame with the 'timestamp:vector' column set as the index.
    """
    # Drop index column
    df.drop(columns=[df.columns[0]], inplace=True)
    # Use timestamp as index
    df['timestamp:vector'] = pd.to_timedelta(df['timestamp:vector'], unit='S')
    df.set_index('timestamp:vector', inplace=True, drop=False)
    return df


def aggregate_by_timedelta(df, timedelta):
    """
    This function aggregates data in a DataFrame based on a given timedelta. It performs the following steps:
    1. Determines the minimum and maximum timestamps in the DataFrame.
    2. Creates a range of timestamps from the minimum to the maximum with a frequency of the given timedelta.
    3. Iterates over each unique 'servingCell:vector' in the DataFrame.
    4. For each 'servingCell:vector', it iterates over each unique 'UEid:vector'.
    5. It groups the data by the timestamp bins and calculates the mean for each group.
    6. It appends the resulting DataFrame to a list.
    7. Finally, it concatenates all the DataFrames in the list into a single DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        timedelta (str): The timedelta for grouping the data.

    Returns:
        df (pd.DataFrame): The aggregated DataFrame.
    """
    
    # Bins go from 0 -> end of simulation data by seconds
    min_timestamp = df.index.min()
    max_timestamp = df.index.max()
    time_range = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=timedelta)
    concat_DF = []
    for cellID in df['servingCell:vector'].unique():
        bsDF = df[df['servingCell:vector'] == cellID]
        for UEid in bsDF['UEid:vector'].unique():
            ueDF = bsDF[bsDF['UEid:vector'] == UEid]
            ueDF_bins = pd.cut(ueDF.index, time_range)
            ueDF = ueDF.groupby(ueDF_bins).mean()
            ueDF.dropna(inplace=True)
            concat_DF.append(ueDF)
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)
    return df
    

def set_time_from_interval(df):
    """
    This function takes a DataFrame and sets the timestamp column as the start
    time of the time interval in the index of the dataframe.
    """


def aggregate_across_basestations(
        df,
        fault_start,
        fault_end,
        servingCells,
        ):
    """
    This function aggregates data across base stations. It performs the following steps:
    1. If fault_end is not provided, it sets it to the maximum timestamp in the DataFrame.
    2. If fault_start is not provided, it sets it to 0.0.
    3. Converts fault_start and fault_end to Timedelta.
    4. Iterates over each unique 'servingCell:vector' in the DataFrame.
    5. For each 'servingCell:vector', it groups the data by the index and calculates the mean for each group.
    6. It then labels each row where 'servingCell:vector' is in servingCells and the timestamp is between fault_start and fault_end.
    7. It appends the resulting DataFrame to a list.
    8. Finally, it concatenates all the DataFrames in the list into a single DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        fault_start (float): The start time for labeling. If None, it is set to 0.0.
        fault_end (float): The end time for labeling. If None, it is set to the maximum timestamp in df.
        servingCells (list): The list of serving cells for labeling.

    Returns:
        df (pd.DataFrame): The aggregated DataFrame.
    """
    # Set default values for fault_start and fault_end if not provided
    if fault_end is None:
        fault_end = df.index.max()
    if fault_start is None:
        fault_start = 0.0

    # Convert fault_start and fault_end to Timedelta
    fault_start = pd.Timedelta(fault_start, unit='S')
    fault_end = pd.Timedelta(fault_end, unit='S')
    
    concat_DF = []
    # Iterate over each unique 'servingCell:vector'
    for cellID in df['servingCell:vector'].unique():
        bsDF = df[df['servingCell:vector'] == cellID]
        # Group by index and calculate mean
        bsDF = bsDF.groupby(by=[bsDF.index]).mean()
        # Label rows where 'servingCell:vector' is in servingCells and timestamp is between fault_start and fault_end
        for cell in servingCells:
            bsDF.loc[(bsDF['servingCell:vector'] == cell) & (bsDF.index > fault_start) & (bsDF.index < fault_end), 'label'] = 1
        concat_DF.append(bsDF)
    
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)
    return df


def write_MTGNN_data(df,keep_rows, save_path):

    # Save data for MTGNN
    new_dataframe = pd.DataFrame()
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add SINR values untill 1954 to new dataframe
        new_dataframe[f"{cell_id}_posx"] = cell_df['positionX:vector'].values[:keep_rows]
        new_dataframe[f"{cell_id}_posy"] = cell_df['positionY:vector'].values[:keep_rows]
        new_dataframe[f"{cell_id}_dist"] = cell_df['servingDistance:vector'].values[:keep_rows]
        new_dataframe[f"{cell_id}_rsrp"] = cell_df['servingRSRP:vector'].values[:keep_rows]
        new_dataframe[f"{cell_id}_rsrq"] = cell_df['servingRSRQ:vector'].values[:keep_rows]
        new_dataframe[f"{cell_id}_sinr"] = cell_df['servingSINR:vector'].values[:keep_rows]
        new_dataframe[f"{cell_id}_thro"] = cell_df['rlcThroughputDl:vector'].values[:keep_rows]
        
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add label column for each cell as well
        new_dataframe[f"{cell_id}_label"] = cell_df['label'].values[:keep_rows]

    # save new_dataframe to a comma separated .txt file
    new_dataframe.to_csv(save_path+'calibrated_multi.txt', sep=',', index=False, header=False)


def write_FCN_data(df,keep_rows, save_path):
    # Save data for FCN
    fcn_df = []
    for cell_id in df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # Keep only the first 2509 rows
        cell_df = cell_df[:keep_rows]
        # append to list
        fcn_df.append(cell_df)
    
    # Concatenate all dataframes
    df = pd.concat(fcn_df, ignore_index=False)
    # Remove timestamp column
    #epr_df.drop(columns=['timestamp:vector','servingCell:vector'], inplace=True)
    df.drop(columns=['servingCell:vector'], inplace=True)

    # Ignore index while saving to csv
    df.to_csv(save_path + "data_FCN.csv", index=False)

def set_time_from_interval(df):
    """
    This function takes a DataFrame and sets the timestamp column as the start
    time of the time interval in the index of the dataframe.
    """
    df.index = df.index.map(lambda x: x.left)
    df.index = pd.to_timedelta(df.index)
    df['timestamp:vector'] = df.index


def prepare_data():
    save_path = "../data/prepared/"
    data_path = "../data/calibrated/EPR686.csv"
    fault_start = 50.0
    fault_end = None
    servingCells = [7, 8]
    keep_rows = 971

    # Read data
    epr_df = pd.read_csv(data_path)

    # Set timestamp as index
    epr_df = set_time_index(epr_df)

    # Aggregate data by 0.1 seconds for user equipments
    epr_df = aggregate_by_timedelta(epr_df, timedelta='.1S')

    epr_df = set_time_from_interval(epr_df)
    
    # Add column label and set all to zero
    epr_df['label'] = 0

    epr_df = aggregate_across_basestations(
        epr_df,
        fault_start=fault_start,
        fault_end=fault_end,
        servingCells=servingCells,
        )

    # Remove UEid column
    epr_df.drop(columns=['UEid:vector'], inplace=True)

    # Reset index to integer values
    epr_df.reset_index(drop=True, inplace=True)
    
    # Count the number of rows for each servingCell
    print(epr_df['servingCell:vector'].value_counts())

    # Drop rows with servingCell 6.0 and 9.0
    epr_df = epr_df[epr_df['servingCell:vector'] != 6.0]
    epr_df = epr_df[epr_df['servingCell:vector'] != 9.0]

    # Write data for MTGNN
    write_MTGNN_data(epr_df,keep_rows, save_path)
    
    # Write data for FCN
    write_FCN_data(epr_df,keep_rows, save_path)


if __name__ == "__main__":
    #prepare_fcn_data()
    prepare_data()
