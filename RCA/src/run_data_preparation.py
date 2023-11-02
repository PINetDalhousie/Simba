from prepare_data import PrepareData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# Returns a dictionary of dataframes with average metrics for each base station
def dfs_by_base_station(df):
    baseStationDFs = {}

    # Bins go from 0 -> end of simulation data by seconds
    min_timestamp = pd.Timedelta(0)
    max_timestamp = df.index.max()
    time_range = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq='S')

    # Get a dataframe for every unique servingCell ID
    for cellID in df['servingCell:vector'].unique():

        # Can remove this when we figure out the non-integer values
        if cellID.is_integer():

            # Generate a dataframe for each base station ID where metrics are averaged in each bin
            bsDF = df[df['servingCell:vector'] == cellID]
            bsDF_bins = pd.cut(bsDF.index, time_range)
            bsDF = bsDF.groupby(bsDF_bins).mean()
            baseStationDFs[cellID] = bsDF

    return baseStationDFs


# Returns a dataframe where basestation dataframes are joined by their timestamps
def concat_base_stations(bsDFs):

    # Get first base station and add label to columns
    cellIDs = iter(bsDFs)
    first_id = next(cellIDs)
    concat_df = bsDFs[first_id]
    concat_df = concat_df.add_suffix(f'_{first_id}')
    print(first_id)

    # Merge the rest of the base stations by index (should be timestamp bins)
    for id in cellIDs:
        print(id)
        concat_df = pd.merge(
            concat_df, bsDFs[id].add_suffix(f'_{id}'), left_index=True, right_index=True)

    return concat_df


# Returns a dataframe where basestation dataframes are joined by their timestamps
def concat_base_stations(bsDFs):

    # Get first base station and add label to columns
    cellIDs = iter(bsDFs)
    first_id = next(cellIDs)
    concat_df = bsDFs[first_id]
    concat_df = concat_df.add_suffix(f'_{first_id}')
    print(first_id)

    # Merge the rest of the base stations by index (should be timestamp bins)
    for id in cellIDs:
        print(id)
        concat_df = pd.merge(
            concat_df, bsDFs[id].add_suffix(f'_{id}'), left_index=True, right_index=True)

    return concat_df


def prepare_data():
    save_path = "../data/"

    # Read data
    epr_df = pd.read_csv('../data/calibrated/EPR686.csv')

    # Drop index and columns we aren't using
    epr_df.drop(columns=[epr_df.columns[0]], inplace=True)

    # Print dataframe info
    #print(epr_df.info(verbose=True))

    # For servingCell 7 or 8 and timestamp greater than 50 set label to 1
    #epr_df.loc[(epr_df['servingCell:vector'] == 7) & (epr_df['timestamp:vector'] > 50), 'label'] = 1
    #epr_df.loc[(epr_df['servingCell:vector'] == 8) & (epr_df['timestamp:vector'] > 50), 'label'] = 1

    # Use timestamp as index
    epr_df['timestamp:vector'] = pd.to_timedelta(epr_df['timestamp:vector'], unit='S')
    epr_df.set_index('timestamp:vector', inplace=True, drop=False)

    # Bins go from 0 -> end of simulation data by seconds
    min_timestamp = epr_df.index.min()
    #min_timestamp = pd.Timedelta(0)
    max_timestamp = epr_df.index.max()
    time_range = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq='.1S')

    concat_DF = []
    # Get a dataframe for every unique servingCell ID
    for cellID in epr_df['servingCell:vector'].unique():
        bsDF = epr_df[epr_df['servingCell:vector'] == cellID]
        # Get a dataframe for every unique servingCell ID
        for UEid in bsDF['UEid:vector'].unique():
            ueDF = bsDF[bsDF['UEid:vector'] == UEid]
            ueDF_bins = pd.cut(ueDF.index, time_range)
            ueDF = ueDF.groupby(ueDF_bins).mean()
            concat_DF.append(ueDF)
            # Remove nan rows
            ueDF.dropna(inplace=True)
    
    # Concatenate all dataframes
    epr_df = pd.concat(concat_DF, ignore_index=False)

    # Save csv
    epr_df.to_csv(save_path + "test_before.csv", index_label='timestamp')

    # Add column label and set all to zero
    epr_df['label'] = 0

    # Groupby and mean with same servingCell and timestamp
    concat_DF = []
    # Get a dataframe for every unique servingCell ID
    for cellID in epr_df['servingCell:vector'].unique():
        bsDF = epr_df[epr_df['servingCell:vector'] == cellID]
        # Groupby timestamp:vector and get mean across features and keep the timestamp column
        bsDF = bsDF.groupby(by=['timestamp:vector']).mean()

        # Initialize pd.Timedelta to 50 seconds
        fifty_seconds = pd.Timedelta(50.0, unit='S')

        # If servingCell is 7 and timestamp is greater than 50 seconds set label to 1
        bsDF.loc[(bsDF['servingCell:vector'] == 7) & (bsDF.index > fifty_seconds), 'label'] = 1
        # If servingCell is 8 and timestamp is greater than 50 set label to 1
        bsDF.loc[(bsDF['servingCell:vector'] == 8) & (bsDF.index > fifty_seconds), 'label'] = 1

        bsDF['timestamp:vector'] = bsDF.index
        concat_DF.append(bsDF)
        # Remove nan rows
        #bsDF.dropna(inplace=True)
     
    # Concatenate all dataframes
    epr_df = pd.concat(concat_DF, ignore_index=False)

    # Add column timestamp:vector
    epr_df['timestamp:vector'] = epr_df.index

    # Remove UEid column
    epr_df.drop(columns=['UEid:vector'], inplace=True)

    # Reset index to integer values
    epr_df.reset_index(drop=True, inplace=True)
    
    # Count the number of rows for each servingCell
    print(epr_df['servingCell:vector'].value_counts())

    # Drop rows with servingCell 6.0
    epr_df = epr_df[epr_df['servingCell:vector'] != 6.0]
    epr_df = epr_df[epr_df['servingCell:vector'] != 9.0]
    
    # Save data for MTGNN
    # create a new dataframe where each column is SINR values of one cell id
    new_dataframe = pd.DataFrame()

    ROWS_TO_KEEP = 2455

    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in epr_df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = epr_df[epr_df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add SINR values untill 1954 to new dataframe
        new_dataframe[f"{cell_id}_posx"] = cell_df['positionX:vector'].values[:ROWS_TO_KEEP]
        new_dataframe[f"{cell_id}_posy"] = cell_df['positionY:vector'].values[:ROWS_TO_KEEP]
        new_dataframe[f"{cell_id}_dist"] = cell_df['servingDistance:vector'].values[:ROWS_TO_KEEP]
        new_dataframe[f"{cell_id}_rsrp"] = cell_df['servingRSRP:vector'].values[:ROWS_TO_KEEP]
        new_dataframe[f"{cell_id}_rsrq"] = cell_df['servingRSRQ:vector'].values[:ROWS_TO_KEEP]
        new_dataframe[f"{cell_id}_sinr"] = cell_df['servingSINR:vector'].values[:ROWS_TO_KEEP]
        new_dataframe[f"{cell_id}_thro"] = cell_df['rlcThroughputDl:vector'].values[:ROWS_TO_KEEP]
        
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in epr_df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = epr_df[epr_df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add label column for each cell as well
        new_dataframe[f"{cell_id}_label"] = cell_df['label'].values[:ROWS_TO_KEEP]

    # save new_dataframe to a comma separated .txt file
    new_dataframe.to_csv('../MTGNN/data/calibrated_multi.txt', sep=',', index=False, header=False)

    # Save data for FCN
    fcn_df = []
    for cell_id in epr_df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = epr_df[epr_df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # Keep only the first 2509 rows
        cell_df = cell_df[:ROWS_TO_KEEP]
        # append to list
        fcn_df.append(cell_df)
    
    # Concatenate all dataframes
    epr_df = pd.concat(fcn_df, ignore_index=False)
    # Remove timestamp column
    epr_df.drop(columns=['timestamp:vector','servingCell:vector'], inplace=True)
    # Ignore index while saving to csv
    epr_df.to_csv(save_path + "data_FCN.csv", index=False)
    
    epr_run = dfs_by_base_station(epr_df)

    #normal_concat = concat_base_stations(normal_run)
    epr_concat = concat_base_stations(epr_run)
    epr_concat.to_csv(save_path + "test.csv", index_label='timestamp')


def prepare_fcn_data():
    prepare_data = PrepareData()

    prepare_data.read_data(
        '../data/latest/'
        )
    
    # cast columns to int
    prepare_data.cast_cell_to_int()
    prepare_data.cast_timestamp_to_int()

    # aggregate using cell id and timestamp
    prepare_data.aggregate()

    # label data
    prepare_data.label_normal(class_id=0)
    prepare_data.label_epr(cell_id=2,class_id=1)
    prepare_data.label_interference(cell_id=5,class_id=1)

    # combine dataframes
    combined_df = prepare_data.merge()

    # remove all columns except for servingcell,timestamp and SINR
    keep_columns = [
        'servingCell:vector',
        'timestamp:vector',
        'servingSINR:vector',
        'servingRSRP:vector',
        'label'
        ]
    # drop all other columns
    combined_df = combined_df.drop(columns=[col for col in combined_df.columns if col not in keep_columns])

    # create a new dataframe where each column is SINR values of one cell id
    new_dataframe = pd.DataFrame()

    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in combined_df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = combined_df[combined_df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add SINR values untill 161098 to new dataframe
        new_dataframe[f"{cell_id}_sinr"] = cell_df['servingSINR:vector'].values[:161098]
        new_dataframe[f"{cell_id}_rsrp"] = cell_df['servingRSRP:vector'].values[:161098]
        
        
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in combined_df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = combined_df[combined_df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add label column for each cell as well
        new_dataframe[f"{cell_id}_label"] = cell_df['label'].values[:161098]

    # save new_dataframe to a comma separated .txt file
    new_dataframe.to_csv('../MTGNN/data/sinr_label_multi.txt', sep=',', index=False, header=False)


if __name__ == "__main__":
    #prepare_fcn_data()
    prepare_data()
