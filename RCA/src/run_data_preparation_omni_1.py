from prepare_data import PrepareData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import datetime
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Get current time
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_faults_df_omni(fault_description_txt:str) -> pd.DataFrame:
    """
    This function reads a fault description text file
    and returns a DataFrame with the fault information.
    """
    # Read the fault description text file into a DataFrame
    # Use the first line as header
    df = pd.read_csv(fault_description_txt, sep=',')    

    # Set dtype for individual columns
    df['fault'] = df['fault'].astype('category')
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['bs'] = df['bs'].astype(int)

    return df

def get_faults_df(fault_description_txt:str) -> pd.DataFrame:
    """
    This function reads a fault description text file, processes each line, and returns a DataFrame with the fault information.
    
    Parameters:
    fault_description_txt (str): The path to the fault description text file.

    Returns:
    df (DataFrame): A DataFrame containing the fault information. Each row corresponds to a fault, with columns for 'Fault Type', 'Start Time', 'End Time', and 'Base Station'.
    """
    # Initialize an empty list to store the fault data
    data = []

    # Open the input file
    with open(fault_description_txt, 'r') as infile:        
        # Process each line in the input file
        for line in infile:
            # Split the line into components
            components = line.strip().split()
            
            # Extract the fault type, start time, end time, and base station from the components
            # The start time and end time have an 's' at the end which needs to be removed
            # The base station is an integer value
            fault_type = components[0]
            start_time = components[1].split('=')[1][:-2]
            end_time = components[2].split('=')[1][:-2]
            base_station = components[4].split('=')[1]
            
            # Create a dictionary with the fault information and add it to the data list
            data.append({'Fault Type': fault_type, 'Start Time': start_time, 'End Time': end_time, 'Base Station': base_station})

    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)

    # Set dtype for individual columns
    df['Fault Type'] = df['Fault Type'].astype('category')
    df['Start Time'] = df['Start Time'].astype(int)
    df['End Time'] = df['End Time'].astype(int)
    df['Base Station'] = df['Base Station'].astype(int)

    return df


def gnb_to_id_mapping(gnb_txt):
    """
    This function reads a text file and creates a mapping from 'gnb' values to corresponding 'ID' values.
    The 'gnb' values are obtained by removing the last digit from the 'gnb' values in the file.
    The 'ID' values are collected into a list for each unique 'gnb' value.

    Args:
        gnb_txt (str): The path to the input text file.

    Returns:
        dict: A dictionary where the keys are 'gnb' values and the values are lists of 'ID' values.
    """
    # Initialize an empty dictionary for the mapping
    mapping = {}

    # Open the input file for reading
    with open(gnb_txt, 'r') as infile:
        # Skip the header line of the file
        next(infile)

        # Process each line in the input file
        for line in infile:
            # Split the line into components and strip leading/trailing whitespace
            components = line.strip().split()

            # Extract the 'gnb' value (without the last digit) and the 'ID' value
            gnb = int(components[1][:-1])
            id = int(components[3])

            # Add the 'ID' value to the list for this 'gnb' value in the mapping
            # If the 'gnb' value is not already a key in the mapping, add it
            mapping.setdefault(gnb, []).append(id)

    return mapping

def get_fault_periods_omni(
        fault_description_df:pd.DataFrame,
        map_fault_to_label:dict,
        ):
    """
    This function takes a DataFrame containing fault descriptions and a mapping dictionary.
    It then maps the fault types to their corresponding labels.
    """
    # Map 'fault' to its corresponding label using the provided dictionary
    fault_description_df['fault'] = fault_description_df['fault'].map(map_fault_to_label)

    # Extract the values of 'Fault Type', 'Start Time', 'End Time', and 'Base Station' into lists
    fault_label = fault_description_df['fault'].values.tolist()
    fault_start = fault_description_df['start'].values.tolist()
    fault_end = fault_description_df['end'].values.tolist()
    servingCells = fault_description_df['bs'].values.tolist()

    # Return the four lists
    return fault_label, fault_start, fault_end, servingCells

def get_fault_periods(
        fault_description_df:pd.DataFrame,
        map_fault_to_label:dict,
        map_bs_to_int:dict,
        ):
    """
    This function takes a DataFrame containing fault descriptions and two mapping dictionaries. 
    It then maps the fault types and base stations to their corresponding labels and integers respectively.
    
    Parameters:
    fault_description_df (pd.DataFrame): A DataFrame containing fault descriptions. 
    It is expected to have the following columns:
    - 'Fault Type': The type of fault.
    - 'Base Station': The base station where the fault occurred.
    - 'Start Time': The start time of the fault.
    - 'End Time': The end time of the fault.
    
    map_fault_to_label (dict): A dictionary mapping fault types to labels.
    
    map_bs_to_int (dict): A dictionary mapping base stations to integers.
    
    Returns:
    tuple: A tuple containing four lists:
    - fault_label: The labels corresponding to the fault types.
    - fault_start: The start times of the faults.
    - fault_end: The end times of the faults.
    - servingCells: The integers corresponding to the base stations.
    """
    
    # Map 'Fault Type' to its corresponding label using the provided dictionary
    fault_description_df['Fault Type'] = fault_description_df['Fault Type'].map(map_fault_to_label)
    
    # Map 'Base Station' to its corresponding integer using the provided dictionary
    fault_description_df['Base Station'] = fault_description_df['Base Station'].map(map_bs_to_int)

    # Extract the values of 'Fault Type', 'Start Time', 'End Time', and 'Base Station' into lists
    fault_label = fault_description_df['Fault Type'].values
    fault_start = fault_description_df['Start Time'].values
    fault_end = fault_description_df['End Time'].values
    servingCells = fault_description_df['Base Station'].values

    # Return the four lists
    return fault_label, fault_start, fault_end, servingCells



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
    # Use timestamp as index
    df['timestamp:vector'] = pd.to_timedelta(df['timestamp:vector'], unit='S')
    df.set_index('timestamp:vector', inplace=True, drop=False)
    return df

# def aggregate_by_timedelta_omni(df:pd.DataFrame, timedelta):
#     # Set min timestamp to 0
#     min_timestamp = pd.Timedelta('0 days 00:00:00')
#     max_timestamp = df.index.max()
#     time_range = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=timedelta)
#     concat_DF = []


#     for cellID in df['servingCell:vector'].unique():
#         bsDF = df[df['servingCell:vector'] == cellID].copy(deep=True)
#         bsDF_bins = pd.cut(bsDF.index, time_range)
#         bsDF = bsDF.groupby(bsDF_bins).mean()
#         bsDF.dropna(inplace=True)
#         concat_DF.append(bsDF)

#     # Concatenate all dataframes
#     df = pd.concat(concat_DF, ignore_index=False)
#     return df


def aggregate_by_timedelta(
        df:pd.DataFrame,
        timedelta,
        servingCell_column:str='servingCell:vector',
        UEid_column:str='UEid',
        ):
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
    #min_timestamp = df['timestamp:vector'].min()
    # Set min timestamp to 0
    min_timestamp = pd.Timedelta('0 days 00:00:00')
    max_timestamp = df.index.max()
    time_range = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=timedelta)
    concat_DF = []
    for cellID in df[servingCell_column].unique():
        bsDF = df[df[servingCell_column] == cellID].copy(deep=True)
        # Save to csv before
        #bsDF.to_csv(f"../data/prepared/{CURRENT_TIME}_bs{cellID}_before_ue.csv", index=False)
        for UEid in bsDF[UEid_column].unique():
            ueDF = bsDF[bsDF[UEid_column] == UEid].copy(deep=True)
            ueDF_bins = pd.cut(ueDF.index, time_range)
            ueDF = ueDF.groupby(ueDF_bins).mean()
            ueDF.dropna(inplace=True)
            concat_DF.append(ueDF)
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)
    # Save to csv after
    #df.to_csv(f"../data/prepared/{CURRENT_TIME}_bs_5_after_ue.csv", index=False)
    #print(asd)
    return df
    

def set_time_from_interval(df):
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


def aggregate_across_basestations_old(
        df:pd.DataFrame,
        fault_start:float,
        fault_end:float,
        servingCells:list,
        fault_labels:list,
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
        fault_labels (list): The list of fault labels for labeling.

    Returns:
        df (pd.DataFrame): The aggregated DataFrame.
    """
    # Set default values for fault_start and fault_end if not provided
    # if fault_end is None:
    #     fault_end = df.index.max()
    # if fault_start is None:
    #     fault_start = 0.0
    
    concat_DF = []

    bsIds = list(df['servingCell:vector'].unique())
    # Sort the list
    bsIds.sort()
    #print(f"cells list {bsIds}")

    # Iterate over each unique 'servingCell:vector'
    for cellID in bsIds:
        # Make a deep copy of bsDF when df['servingCell:vector'] == cellID
        bsDF = df[df['servingCell:vector'] == cellID].copy(deep=True)

        # Group by index
        bsDF_group = bsDF.groupby(by=bsDF['time_milli'])

        # Calculate mean of each group when grouped by timestamp:vector
        bsDF = bsDF_group.mean() #numeric_only=True)

        # Add a column called 'count' that contains the number of rows in each group
        bsDF['count'] = bsDF_group.size()
        
        # Add labels to rows where 'servingCell:vector' is in servingCells and the timestamp is between fault_start and fault_end
        for start,end,cells,fault_label in zip(fault_start,fault_end,servingCells,fault_labels):
            if cells != cellID:
                continue
            # Convert fault_start and fault_end to Timedelta
            start = pd.Timedelta(start, unit='S')
            end = pd.Timedelta(end, unit='S')
            print(start)
            print(end)
            print(cells)
            print(fault_label)
            # # Print number of unique celss
            # print(bsDF['servingCell:vector'].value_counts())
            # print(asd)
            
            #for cell in cells:
            #bsDF.loc[(bsDF['servingCell:vector'] == float(cells)) & (bsDF['timestamp:vector'] >= start) & (bsDF['timestamp:vector'] <= end), 'label'] = fault_label
            bsDF.loc[(bsDF['timestamp:vector'] >= start) & (bsDF['timestamp:vector'] <= end), 'label'] = fault_label
        print(f"bs {cellID}, count {bsDF['label'].value_counts()}")


        concat_DF.append(bsDF)
    
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)

    return df


def aggregate_across_basestations(
        df:pd.DataFrame,
        fault_start:float,
        fault_end:float,
        servingCells:list,
        fault_labels:list,
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
        fault_labels (list): The list of fault labels for labeling.

    Returns:
        df (pd.DataFrame): The aggregated DataFrame.
    """
    # Create a faults df with the fault start and end times
    faults_df = pd.DataFrame({
        'fault_start': fault_start,
        'fault_end': fault_end,
        'servingCells': servingCells,
        'fault_labels': fault_labels,
    })
    # Set start and end as pd.timedelta seconds
    faults_df['fault_start'] = pd.to_timedelta(faults_df['fault_start'], unit='S')
    faults_df['fault_end'] = pd.to_timedelta(faults_df['fault_end'], unit='S')

    
    concat_DF = []

    bsIds = list(df['servingCell:vector'].unique())
    # Sort the list
    bsIds.sort()
    #print(f"cells list {bsIds}")

    # Iterate over each unique 'servingCell:vector'
    for cellID in bsIds:
        # Make a deep copy of bsDF when df['servingCell:vector'] == cellID
        bsDF = df[df['servingCell:vector'] == cellID].copy(deep=True)

        # Group by index
        bsDF_group = bsDF.groupby(by=bsDF['time_milli'])

        # Calculate mean of each group when grouped by timestamp:vector
        bsDF = bsDF_group.mean() #numeric_only=True)

        # Add a column called 'count' that contains the number of rows in each group
        bsDF['count'] = bsDF_group.size()
        
        # Get rows from faults_df where servingCells is equal to cellID
        faults_df_cell = faults_df[faults_df['servingCells'] == cellID]

        # Iterate over the rows of bsDF
        for index, row in bsDF.iterrows():
            # Check if current row timestamp is greater than or equal to any fault start time in faults_df_cell
            # and less than or equal to any fault end time in faults_df_cell
            # To compare time, use bsDF timestamp:vector column
            for i, fault_row in faults_df_cell.iterrows():
                if (row["timestamp:vector"] >= fault_row['fault_start']) and (row["timestamp:vector"] <= fault_row['fault_end']):
                    # Set the label of the current row to the fault label
                    bsDF.at[index, 'label'] = fault_row['fault_labels']
                    break

        print(f"bs {cellID}, count {bsDF['label'].value_counts()}")
        # Save to csv before
        bsDF.to_csv(f"../data/prepared/{CURRENT_TIME}_bs{cellID}_before_ue.csv", index=False)


        # # Add labels to rows where 'servingCell:vector' is in servingCells and the timestamp is between fault_start and fault_end
        # for start,end,cells,fault_label in zip(fault_start,fault_end,servingCells,fault_labels):
        #     if cells != cellID:
        #         continue
        #     # Convert fault_start and fault_end to Timedelta
        #     start = pd.Timedelta(start, unit='S')
        #     end = pd.Timedelta(end, unit='S')
        #     print(start)
        #     print(end)
        #     print(cells)
        #     print(fault_label)
        #     # # Print number of unique celss
        #     # print(bsDF['servingCell:vector'].value_counts())
        #     # print(asd)
            
        #     #for cell in cells:
        #     #bsDF.loc[(bsDF['servingCell:vector'] == float(cells)) & (bsDF['timestamp:vector'] >= start) & (bsDF['timestamp:vector'] <= end), 'label'] = fault_label
        #     bsDF.loc[(bsDF['timestamp:vector'] >= start) & (bsDF['timestamp:vector'] <= end), 'label'] = fault_label
        # print(f"bs {cellID}, count {bsDF['label'].value_counts()}")


        concat_DF.append(bsDF)
    
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)

    return df


def write_MTGNN_data(df, save_path): #, keep_rows):
    """
    This function prepares data for MTGNN (Multi-Time-Graph Neural Network) and saves it to a .txt file.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    save_path (str): The path where the output .txt file will be saved.
    keep_rows (int): The number of rows to keep from each unique cell in the dataframe.

    Returns:
    None
    """

    # Get list of unique bast station ids
    bs_ids = list(df['servingCell:vector'].unique())
    # Sort the list
    bs_ids.sort()

    # Save data for MTGNN
    #new_dataframe = pd.DataFrame()
    data = {}
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in bs_ids:
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index to have integer values
        cell_df = cell_df.reset_index(drop=True)
        
        # add SINR values untill 1954 to new dataframe
        data[f"{cell_id}_posx"] = list(cell_df['positionX:vector'].values) #[:keep_rows]
        data[f"{cell_id}_posy"] = list(cell_df['positionY:vector'].values) #[:keep_rows]
        data[f"{cell_id}_dist"] = list(cell_df['servingDistance:vector'].values) #[:keep_rows]
        data[f"{cell_id}_rsrp"] = list(cell_df['servingRSRP:vector'].values) #[:keep_rows]
        data[f"{cell_id}_rsrq"] = list(cell_df['servingRSRQ:vector'].values) #[:keep_rows]
        data[f"{cell_id}_sinr"] = list(cell_df['servingSINR:vector'].values) #[:keep_rows]
        #data[f"{cell_id}_thro"] = list(cell_df['rlcThroughputDl:vector'].values)[:keep_rows]

    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in bs_ids:
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add label column for each cell as well
        data[f"{cell_id}_label"] = list(cell_df['label'].values) #[:keep_rows]

    # Create new dataframe from dictionary
    new_dataframe = pd.DataFrame.from_dict(data)

    # save new_dataframe to a comma separated .txt file
    new_dataframe.to_csv(save_path+f'{CURRENT_TIME}_MTGNN.txt', sep=',', index=False, header=False)


def write_FCN_data(df, save_path): #, keep_rows):
    """
    This function prepares data for FCN (Fully Connected Network) and saves it to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    save_path (str): The path where the output CSV file will be saved.
    keep_rows (int): The number of rows to keep from each unique cell in the dataframe.

    Returns:
    None
    """
    # Save data for FCN
    fcn_df = []
    for cell_id in df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # Keep only the first 2509 rows
        #cell_df = cell_df #[:keep_rows]
        # append to list
        fcn_df.append(cell_df)
    
    # Concatenate all dataframes
    df = pd.concat(fcn_df, ignore_index=False)
    # Remove timestamp column
    df.drop(columns=['servingCell:vector'], inplace=True)

    # Ignore index while saving to csv
    df.to_csv(save_path + f"{CURRENT_TIME}_base.csv", index=False)

def set_time_from_interval(df):
    """
    This function takes a DataFrame and sets the timestamp column as the start
    time of the time interval in the index of the dataframe.
    """
    df.index = df.index.map(lambda x: x.left)
    df.index = pd.to_timedelta(df.index)
    #df['timestamp:vector'] = df.index
    return df


def plot_cell_kpi_vs_time(df, cell_id, kpis, ouput_path):
    """
    This function plots the given KPIs for the given cell ID against time.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cell_id (int): The cell ID to plot.
        kpis (list): The list of KPIs to plot.
    """
    # Get the dataframe for the given cell ID
    cell_df = df[df['servingCell:vector'] == cell_id]

    # Reset index to integer values
    cell_df.reset_index(drop=True, inplace=True)

    # Convert timstamp to seconds as integers
    cell_df['timestamp:vector'] = cell_df['timestamp:vector'].dt.total_seconds().astype(int)

    # Sort by timestamp
    cell_df = cell_df.sort_values(by='timestamp:vector')

    # Plot the given KPIs against time
    for kpi in kpis:
        plt.plot(cell_df['timestamp:vector'], cell_df[kpi])
        plt.xlabel('Time')
        plt.ylabel(kpi)
        plt.show()
        plt.savefig(f"{ouput_path}/{cell_id}_{kpi}.png")
        plt.close()


def aggregate_paired_basestations(df):
    """
    This function aggregates data from paired base stations.

    It first resets the index of the DataFrame to integer values. Then, it reduces the 'servingCell:vector' values by 1.0 if they are even.
    After that, it groups the DataFrame by 'servingCell:vector' and 'timestamp:vector' and calculates the mean of the grouped data.
    It then resets the index to move 'servingCell:vector' and 'timestamp:vector' back to the columns.
    Finally, it sets 'timestamp:vector' as the index of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    pandas.DataFrame: The processed DataFrame with aggregated data.
    """
    # Reset index to integer values
    df.reset_index(drop=True, inplace=True)
    # Reduce servingCell:vector values by 1.0 if they are even
    df.loc[df['servingCell:vector'] % 2 == 0, 'servingCell:vector'] = df['servingCell:vector'] - 1.0    
    # Group by servingCell:vector and timestep and calculate the mean
    df = df.groupby(['servingCell:vector', 'timestamp:vector'],group_keys=False).mean()
    # Reset the index to move servingCell:vector and timestamp:vector back to the columns
    df = df.reset_index()
    # set timestamp:vector column as index
    # Use timestamp as index
    df.set_index('timestamp:vector', inplace=True, drop=False)
    return df


def prepare_data():
    MAP_FAULT_TO_LABEL = {
        'TLHO': 1,
        'INTERFERENCE' : 2,
        'EPR' : 3,
    }
    TIME_INTERVAL = 1.0
    KEEP_ROWS = 3572

    # Parse arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="../data/large_topology/", 
                        help='The path to save the prepared data.')
    parser.add_argument('--data_path', type=str, default="../data/large_topology/large-topo-simultaneous-v1.csv",
                        help='The path to the input data.')
    parser.add_argument('--fault_description_txt', type=str, default="../data/large_topology/fault-description.txt",
                        help='The path to the fault-description.txt file.')
    args = parser.parse_args()


    ############################
    # Read description file
    ############################
    fault_description_df = get_faults_df_omni(args.fault_description_txt)
    
    #print(fault_description_df)
    #print(asd)
    # Get fault description dataframe
    fault_label, fault_start, fault_end, servingCells = get_fault_periods_omni(
        fault_description_df,
        MAP_FAULT_TO_LABEL,
        )
    
    # print(fault_label)
    # print(fault_start)
    # print(fault_end)
    # print(servingCells)
    # # Print length of these
    # print(len(fault_label))
    # print(len(fault_start))
    # print(len(fault_end))
    # print(len(servingCells))
    # print(asd)

    
    ############################
    # Read data file csv
    ############################
    # Read time series kpi data
    df = pd.read_csv(args.data_path)



    ############################
    # Filter unnecessary Time columns
    ############################
    # Rename the column "positionX:vector-Time" as 'timestamp:vector'
    df.rename(columns={'positionX:vector-Time':'timestamp:vector'}, inplace=True)
    # Drop columns with 'Time' at the end 
    df = df.loc[:,~df.columns.str.contains('Time')]
    # Drop index column
    df.drop(columns=[df.columns[0]], inplace=True)
    df = set_time_index(df)
    print(f"Number of rows before aggregation: {len(df)}")
    


    ############################
    # Aggregate data by seconds for user equipments
    ############################
    df = aggregate_by_timedelta(df, timedelta=f'{TIME_INTERVAL}S')
    print(f"Number of rows after aggregation: {len(df)}")
    df = set_time_from_interval(df)
    df['servingCell:vector'] = df['servingCell:vector'].astype(int)
    df['timestamp:vector'] = df.index
    df['time_milli'] = df['timestamp:vector'].dt.total_seconds().astype(float)
    # Reset index to integer values
    df.reset_index(drop=True, inplace=True)
    # Remove ueid column
    df.drop(columns=['UEid'], inplace=True)
    print(f"after time bin aggregation : {df.info()}")

    # Print first 50 values from timestamp column for bs 1
    #print(df[df['servingCell:vector'] == 1]['timestamp:vector'].head(50))
    #print(asd)

    ############################
    # Aggregate data by time for base stations
    ############################
    # Aggregate data across base stations
    df = aggregate_across_basestations(
        df,
        fault_start=fault_start,
        fault_end=fault_end,
        servingCells=servingCells,
        fault_labels=fault_label,
        )
    print(f"after ue aggregation : {df.info()}")

    # Set all non labeled rows to label 0
    df['label'].fillna(0, inplace=True)

    ############################
    # Iterate over each cell id and plot the kpis
    ############################
    bsIDs = list(df['servingCell:vector'].unique())
    bsIDs.sort()
    for cell_id in bsIDs:
        plot_cell_kpi_vs_time(
            df, 
            cell_id, 
            [
                'servingDistance:vector',
                'servingRSRP:vector',
                'servingRSRQ:vector',
                'servingSINR:vector',
                #'rlcThroughputDl:vector',
                #'servingDistance:vector',
                'count',
                'label'
            ], 
            f'{args.save_path}/plots',
            )

    print(df.info())


    #print(asd)

    # Set all non-zero label to 1
    #df['label'].replace(to_replace=[1,2,3], value=1, inplace=True)

    print(df['servingCell:vector'].value_counts())
    print(df['label'].value_counts())
    

    ############################
    # Only keep rows for which all basestations have data
    ############################
    # Get sorted list of unique bs
    bsIds = list(df['servingCell:vector'].unique())


    # Sort the list
    bsIds.sort()

    

    # Iterate over the list
    for cellID in bsIds:
        # Get dataframe for current cell id
        cellDF = df[df['servingCell:vector'] == cellID].copy(deep=True)

        if int(cellID) == 1:
            #print(f"getting 1")
            # Create a seperate dataframe from only the timestamp column
            timestampDF = cellDF['timestamp:vector'].copy(deep=True)
            continue
        # Merge the timestamp column with the previous timestamp column
        timestampDF = pd.merge(timestampDF, cellDF['timestamp:vector'], on='timestamp:vector', how='inner')

    # Create a list to store dataframes
    concat_DF = []
    for cellID in bsIds:
        # Get dataframe for current cell id
        cellDF = df[df['servingCell:vector'] == cellID].copy(deep=True)

        # Merge the timestamp column with the previous timestamp column
        cellDF = pd.merge(timestampDF, cellDF, on='timestamp:vector', how='inner')

        concat_DF.append(cellDF)
    
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)
    
    print(f"before writing, bs counts: {df['servingCell:vector'].value_counts()}")
    print(f"before writing, label count: {df['label'].value_counts()}")
    
    #print(asd)
    # Get the minimum from value counts of label. 
    # Only keep rows uptill then
    #MIN_LABEL = min(df['servingCell:vector'].value_counts())

    ############################
    # Write data for MTGNN
    ############################
    write_MTGNN_data(df, args.save_path) #, keep_rows=MIN_LABEL)
    
    ############################
    # Write data for FCN
    ############################
    write_FCN_data(df, args.save_path) #, keep_rows=MIN_LABEL)


if __name__ == "__main__":
    prepare_data()

