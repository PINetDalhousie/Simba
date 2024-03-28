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


def aggregate_by_timedelta(df:pd.DataFrame, timedelta):
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
    min_timestamp = df['timestamp:vector'].min()
    max_timestamp = df['timestamp:vector'].max()
    time_range = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=timedelta)
    concat_DF = []
    for cellID in df['servingCell:vector'].unique():
        bsDF = df[df['servingCell:vector'] == cellID].copy(deep=True)
        # Save to csv before
        bsDF.to_csv(f"../data/prepared/{CURRENT_TIME}_bs{cellID}_before_ue.csv", index=False)
        for UEid in bsDF['UEid:vector'].unique():
            ueDF = bsDF[bsDF['UEid:vector'] == UEid]
            ueDF_bins = pd.cut(ueDF['timestamp:vector'], time_range)
            ueDF = ueDF.groupby(ueDF_bins).mean()
            ueDF.dropna(inplace=True)
            concat_DF.append(ueDF)
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)
    # Save to csv after
    df.to_csv(f"../data/prepared/{CURRENT_TIME}_bs_5_after_ue.csv", index=False)

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
    # Set default values for fault_start and fault_end if not provided
    if fault_end is None:
        fault_end = df.index.max()
    if fault_start is None:
        fault_start = 0.0
    
    concat_DF = []
    # Iterate over each unique 'servingCell:vector'
    for cellID in df['servingCell:vector'].unique():
        # Make a deep copy of bsDF when df['servingCell:vector'] == cellID
        bsDF = df[df['servingCell:vector'] == cellID].copy(deep=True)

        # If cellID is 5 and timstamp is 0 days 00:00:06.004000, save to csv
        if cellID == 5:
            bsDF[bsDF.index == pd.Timedelta('0 days 00:00:06.004000')].to_csv(f"../data/2024-01-24/{CURRENT_TIME}_bs{cellID}_before.csv", index=False)

        # Group by timestamp:vector and calculate mean
        bsDF_group = bsDF.groupby(by=['timestamp:vector'])

        # Calculate mean of each group when grouped by timestamp:vector
        bsDF = bsDF_group.mean(numeric_only=True)

        # If cellID is 5 and timstamp is 0 days 00:00:06.004000, save to csv
        if cellID == 5:
            bsDF[bsDF.index == pd.Timedelta('0 days 00:00:06.004000')].to_csv(f"../data/2024-01-24/{CURRENT_TIME}_bs{cellID}_after.csv", index=False)

        # Calculate the number of rows in each group when grouped by timestamp:vector
        # And add the number of rows in each group as a column
        bsDF['count'] = bsDF_group.size()

        # Save csv
        #bsDF.to_csv(f"../data/prepared/{CURRENT_TIME}_bs{cellID}_grouped.csv", index=False)
        # Label rows where 'servingCell:vector' is in servingCells and timestamp is between fault_start and fault_end
        for start,end,cells,fault_label in zip(fault_start,fault_end,servingCells,fault_labels):
            # Convert fault_start and fault_end to Timedelta
            start = pd.Timedelta(start, unit='S')
            end = pd.Timedelta(end, unit='S')
            for cell in cells:
                bsDF.loc[(bsDF['servingCell:vector'] == cell) & (bsDF.index > start) & (bsDF.index < end), 'label'] = fault_label
        # Save csv
        # bsDF.to_csv(f"../data/prepared/{CURRENT_TIME}_bs{cellID}_labeled.csv", index=False)
        # print(concat_DF)
        # print(asd)
        # print(bsDF.info())
        # print(Asd)
        concat_DF.append(bsDF)
    
    # Concatenate all dataframes
    df = pd.concat(concat_DF, ignore_index=False)

    # Create a new column called 'servingCell:vector' and set it to the index
    df['timestamp:vector'] = df.index

    return df


def write_MTGNN_data(df, save_path, keep_rows):

    # Save data for MTGNN
    #new_dataframe = pd.DataFrame()
    data = {}
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index to have integer values
        cell_df = cell_df.reset_index(drop=True)
        
        # add SINR values untill 1954 to new dataframe
        data[f"{cell_id}_posx"] = list(cell_df['positionX:vector'].values)[:keep_rows]
        data[f"{cell_id}_posy"] = list(cell_df['positionY:vector'].values)[:keep_rows]
        data[f"{cell_id}_dist"] = list(cell_df['servingDistance:vector'].values)[:keep_rows]
        data[f"{cell_id}_rsrp"] = list(cell_df['servingRSRP:vector'].values)[:keep_rows]
        data[f"{cell_id}_rsrq"] = list(cell_df['servingRSRQ:vector'].values)[:keep_rows]
        data[f"{cell_id}_sinr"] = list(cell_df['servingSINR:vector'].values)[:keep_rows]
        data[f"{cell_id}_thro"] = list(cell_df['rlcThroughputDl:vector'].values)[:keep_rows]
        
    # iterate over unique cell ids and retrieve dataframe for each cell id
    for cell_id in df['servingCell:vector'].unique():
        # get dataframe for current cell id
        cell_df = df[df['servingCell:vector']==cell_id]
        # sort by timestamp
        cell_df = cell_df.sort_values(by='timestamp:vector')
        # reset index
        cell_df = cell_df.reset_index(drop=True)
        # add label column for each cell as well
        data[f"{cell_id}_label"] = list(cell_df['label'].values)[:keep_rows]

    # Create new dataframe from dictionary
    new_dataframe = pd.DataFrame.from_dict(data)

    # save new_dataframe to a comma separated .txt file
    new_dataframe.to_csv(save_path+f'{CURRENT_TIME}_MTGNN.txt', sep=',', index=False, header=False)


def write_FCN_data(df, save_path, keep_rows):
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
    df['timestamp:vector'] = df.index
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

    # Plot the given KPIs against time
    for kpi in kpis:
        plt.plot(cell_df['timestamp:vector'], cell_df[kpi])
        plt.xlabel('Time')
        plt.ylabel(kpi)
        plt.savefig(f"{ouput_path}/{cell_id}_{kpi}.png")
        plt.close()

def prepare_data():
    MAP_FAULT_TO_LABEL = {
        'TLHO': 1,
        'INTERFERENCE' : 2,
        'EPR' : 3,
    }
    TIME_INTERVAL = 1.0
    KEEP_ROWS = 1899

    # Parse arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="../data/2024-01-24/", 
                        help='The path to save the prepared data.')
    parser.add_argument('--data_path', type=str, default="../data/2024-01-23/1800s-failure-EPR-TLHO-v1.csv",
                        help='The path to the input data.')
    parser.add_argument('--gnb_map_txt', type=str, default="../data/2024-01-23/gnb-map.txt",
                        help='The path to the gnb-map.txt file.')
    parser.add_argument('--fault_description_txt', type=str, default="../data/2024-01-23/faults-description.txt",
                        help='The path to the fault-description.txt file.')
    args = parser.parse_args()


    ############################
    # Read description file
    ############################
    fault_description_df = get_faults_df(args.fault_description_txt)

    # Get gnb number to base station id mapping
    map_bs_to_int = gnb_to_id_mapping(args.gnb_map_txt)

    # Get fault description dataframe
    fault_label, fault_start, fault_end, servingCells = get_fault_periods(
        fault_description_df,
        MAP_FAULT_TO_LABEL,
        map_bs_to_int,
        )
    
    ############################
    # Read data file csv
    ############################
    # Read time series kpi data
    epr_df = pd.read_csv(args.data_path)

    print(f"epr_df info: {epr_df.info()}")

    # Sample 1000 rows
    #epr_df = epr_df.sample(n=10000, random_state=42)
    
    # Only keep rows with servingCell:vector = 5
    epr_df = epr_df[epr_df['servingCell:vector'] == 5]

    # Set timestamp as index
    epr_df = set_time_index(epr_df)
    print(f"Number of rows before aggregation: {len(epr_df)}")

    # Save to csv file for all rows with servingCell:vector = 5
    #epr_df[epr_df['servingCell:vector'] == 5].to_csv(args.save_path + f"{CURRENT_TIME}_bs5.csv", index=False)


    # Aggregate data by seconds for user equipments
    epr_df = aggregate_by_timedelta(epr_df, timedelta=f'{TIME_INTERVAL}S')
    print(f"Number of rows after aggregation: {len(epr_df)}")

    epr_df = set_time_from_interval(epr_df)

    # Sort by timestamp
    epr_df = epr_df.sort_values(by='timestamp:vector')

    # Save to csv
    #epr_df.to_csv(args.save_path + f"{CURRENT_TIME}_aggr.csv", index=False)

    # Set servingCell:vector to integer
    epr_df['servingCell:vector'] = epr_df['servingCell:vector'].astype(int)

    print(epr_df.info())
    # # Convert all float64 columns to float32
    # for col in epr_df.columns:
    #     if epr_df[col].dtype == 'float64':
    #         epr_df[col] = epr_df[col].astype(np.float32)

    # Aggregate data across base stations
    epr_df = aggregate_across_basestations(
        epr_df,
        fault_start=fault_start,
        fault_end=fault_end,
        servingCells=servingCells,
        fault_labels=fault_label,
        )
    
    print(f"after ue aggregation : {len(epr_df)}")

    print(epr_df.info())

    # Save to csv
    epr_df.to_csv(args.save_path + f"{CURRENT_TIME}_aggr_bs.csv", index=False)
    #print(asd)

    # Iterate over each cell id and plot the kpis
    for cell_id in epr_df['servingCell:vector'].unique():
        plot_cell_kpi_vs_time(
            epr_df, 
            cell_id, 
            [
                'servingDistance:vector',
                'servingRSRP:vector',
                'servingRSRQ:vector',
                'servingSINR:vector',
                'rlcThroughputDl:vector',
                'servingDistance:vector',
                'count',
            ], 
            f'{args.save_path}/plots',
            )

    print(epr_df.info())
    print(asd)



    
    

    print(f"after aggregation: {len(epr_df)}")
    print(epr_df.info())
    print(epr_df['servingCell:vector'].value_counts())
    print(epr_df.head())


    epr_df['label'] = 0
    

    # # Save to csv
    # epr_df.to_csv(save_path + f"{CURRENT_TIME}_temp1.csv", index=False)
    # print(Asd)
    
    print(f"after ue aggregation : {len(epr_df)}")
    print(epr_df.info())
    print(epr_df['servingCell:vector'].value_counts())
    print(epr_df.head())

    # Print value counts for servingCell:vector
    print(epr_df['servingCell:vector'].value_counts())
    # Print value counts for label
    print(epr_df['label'].value_counts())
    #print(asd)

    # # Save to csv
    #epr_df.to_csv(save_path + f"{CURRENT_TIME}_aggr.csv", index=False)

    # Remove UEid column
    epr_df.drop(columns=['UEid:vector'], inplace=True)
    # Reset index to integer values
    epr_df.reset_index(drop=True, inplace=True)
    # Reduce servingCell:vector values by 1.0 if they are even
    epr_df.loc[epr_df['servingCell:vector'] % 2 == 0, 'servingCell:vector'] = epr_df['servingCell:vector'] - 1.0

    # Save to csv
    #epr_df.to_csv(save_path + f"{CURRENT_TIME}_even.csv", index=False)
    
    # Group by servingCell:vector and timestep and calculate the mean
    epr_df = epr_df.groupby(['servingCell:vector', 'timestamp:vector'],group_keys=False).mean()
    # Reset the index to move servingCell:vector and timestamp:vector back to the columns
    epr_df = epr_df.reset_index()
    # set timestamp:vector column as index
    # Use timestamp as index
    epr_df.set_index('timestamp:vector', inplace=True, drop=False)
    # Save to csv
    #epr_df.to_csv(save_path + f"{CURRENT_TIME}_grouped.csv", index=False)

    # Print value counts for servingCell:vector
    print(epr_df['servingCell:vector'].value_counts())
 

    
    # Get the value of servingCell:vector with the lowest number of samples
    # lowest_sampled_bs = epr_df['servingCell:vector'].value_counts().index[-1]
    # print(f"Lowest sampled bs: {lowest_sampled_bs}")
    # # Get the index values for 'timestamp:vector' where 'servingCell:vector' is equal to lowest_sampled_bs
    # timestamps_to_include = epr_df[epr_df['servingCell:vector'] == lowest_sampled_bs].index.get_level_values('timestamp:vector')
    # # Get a boolean mask where each value is True if the 'timestamp:vector' is in timestamps_to_include
    # mask = epr_df.index.get_level_values('timestamp:vector').isin(timestamps_to_include)
    # # Apply the mask to the dataframe to include only the rows with 'timestamp:vector' in timestamps_to_include
    # epr_df = epr_df[mask]

    # # Print value count for label column
    # print(epr_df['label'].value_counts())


    # # Reset index to integer values
    # epr_df.reset_index(drop=True, inplace=True)
    # dataframe = []
    # # Iterate over unique servingCell:vector values
    # for cell_id in epr_df['servingCell:vector'].unique():
    #     # get dataframe for current cell id
    #     cell_df = epr_df[epr_df['servingCell:vector']==cell_id]
    #     # Get missing timestamps that are missing in cell_df['timestamp:vector'] but
    #     # are present in timstamp_to_include
    #     timestamps_to_include_temp = timestamps_to_include[~timestamps_to_include.isin(cell_df['timestamp:vector'])]
    #     # Create empty rows for missing timestamps with timestamp:vector column set to the missing timestamps
    #     # The other columns will be filled with nan values
    #     empty_rows = pd.DataFrame(index=timestamps_to_include_temp, columns=cell_df.columns)
    #     empty_rows['timestamp:vector'] = timestamps_to_include_temp
    #     empty_rows['servingCell:vector'] = cell_id
    #     # Reset index to integer values
    #     empty_rows.reset_index(drop=True, inplace=True)
    #     # Add empty rows to cell_df
    #     cell_df = pd.concat([cell_df,empty_rows])
    #     # sort by timestamp column
    #     cell_df = cell_df.sort_values(by='timestamp:vector')
    #     # Interpolate missing values using linear interpolation
    #     cell_df = cell_df.interpolate(method='linear', limit_direction='both')
    #     # Covert all non zero labels into 1
    #     cell_df.loc[cell_df['label'] != 0, 'label'] = 1
    #     # Print number of nans after interpolation
    #     # Add interpolated dataframe to empty dataframe
    #     dataframe.append(cell_df)


    # Covert all non zero labels into 1
    epr_df.loc[epr_df['label'] != 0, 'label'] = 1

    # # Concatenate all dataframes
    # epr_df = pd.concat(dataframe, ignore_index=False)

    

    # Reset index to integers
    epr_df.reset_index(drop=True, inplace=True)
    
    # save to a csv file
    epr_df.to_csv(save_path + f"{CURRENT_TIME}_final.csv", index=False)
    print(epr_df.info())
    print(epr_df['servingCell:vector'].value_counts())
    print(epr_df['label'].value_counts())
    print(asd)
    # Write data for MTGNN
    write_MTGNN_data(epr_df, save_path, keep_rows)
    
    # Write data for FCNk
    write_FCN_data(epr_df, save_path, keep_rows)


def combine_dataframes(list_of_files):
    pass


if __name__ == "__main__":
    prepare_data()

