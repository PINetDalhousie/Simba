# SIMBA

- Setup two seperate python virtual environments and install all packages from requirements_pandas.txt (for running prepare_data.py) and requirements_train.txt (for running train.py script).
- The data artifacts from simu5G simulation comprises of a csv file and a txt file. 
- The csv contains all the kpi data at the user equipment level and the txt file contains the start and end of each fault.
- The .csv and .txt file serve as the input to prepare_data.py script. Set the appropriate path for your data. Run the prepare_data.py with the right arguments by:
```
python prepare_data.py --save_path "/output/path/of/prepared/data" --data_path "/csv/file/path" --fault_description_txt "/path/to/fault/description/txt"
```
- The prepare_data.py script prepares the data for training into a txt file.
- Use train.py to train a model with prepared data by setting the model argument one of MTGNN or TransformerMTGNN or GCNMTGNN. Set the num_nodes argument to the number of base stations in your dataset. Use:
```
python train.py --model "TransformerMTGNN" --num_nodes 7 --data "/path/to/txt/generated/by/prepare_data/"
```
- Optionally you can tune other parameters for trianing by changing the global variables (in capital) at the start of train.py script.
- Model training will create tensorboard event files under the log directory. Use tensorboard to view trainig and validation metrics.

