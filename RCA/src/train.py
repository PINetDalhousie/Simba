import os
import datetime
import tensorflow as tf
import pandas as pd
from models import FullyConnectedNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from metrics import F1Score
from sklearn.preprocessing import MinMaxScaler


def train_MTGNN():
    #torch.cuda.set_device(0)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.system(f"python ../MTGNN/train_single_step.py --save ../logs/{current_time}.pt --data ../MTGNN/data/solar_AL.txt --num_nodes 137 --batch_size 32 --epochs 30 --horizon 3")

def train_FCN_AD():
    # set training to anomaly detection or root cause analysis
    train_AD = True

    # get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # load csv
    df = pd.read_csv('../data/combined.csv')

    if train_AD:
        # transform for anomaly detection
        # transform all non-zero values of column 'label' to 1
        df['label'] = df['label'].apply(lambda x: 1 if x != 0 else 0)
        # set output size for last layer
        output_size = 2
    else:
        # transform for root cause analysis
        # drop rows with label 0
        df = df[df['label'] != 0]
        # decrease value of non-zero labesl by 1
        df['label'] = df['label'].apply(lambda x: x-1)
        # set output size for last layer
        output_size = 2


    # split into train,validation and test using scikit-learn
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    # save test set
    test.to_csv('../data/test.csv', index=False)

    train_kpis = train.drop(["label"],axis=1)
    min_max_scaler = MinMaxScaler()
    train_kpis = min_max_scaler.fit_transform(train_kpis)

    # cast dataframe to float32
    train_kpis = train_kpis.astype('float32')
    train_labels = pd.get_dummies(train["label"])
    val_kpis = val.drop(["label"],axis=1)
    val_kpis = min_max_scaler.transform(val_kpis)

    # cast dataframe to float32
    val_kpis = val_kpis.astype('float32')
    val_labels = pd.get_dummies(val["label"]) #.astype('int8')
    

    def encode(kpi,label):
        return kpi, tf.one_hot(label,2)

    # Create tf.data.Dataset object from dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_kpis, train_labels))
    #train_ds = train_ds.map(encode)
    train_ds = train_ds.shuffle(10000).batch(32,drop_remainder=True)
    val_ds = tf.data.Dataset.from_tensor_slices((val_kpis, val_labels))
    #train_ds = train_ds.map(encode)
    val_ds = val_ds.shuffle(10000).batch(32,drop_remainder=True) 

    # Create an instance of the model
    model = FullyConnectedNN(output_size=output_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compile the model with binary_crossentropy loss
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[
            tf.keras.metrics.Precision(name='precision_0',class_id=0),
            tf.keras.metrics.Recall(name='recall_0',class_id=0),
            F1Score(name='f1score_0',class_id=0),
            tf.keras.metrics.Precision(name='precision_1',class_id=1),
            tf.keras.metrics.Recall(name='recall_1',class_id=1),
            F1Score(name='f1score_1',class_id=1)
            ])

    # Set up the callbacks
    tensorboard_callback = TensorBoard(log_dir=f'../logs/{current_time}')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'../logs/{current_time}',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
        )

    # Train the model
    model.fit(x=train_ds,
        epochs=100,
        verbose=2,
        callbacks=[
            tensorboard_callback,
            model_checkpoint_callback
            ],
        initial_epoch=0,
        validation_data=val_ds,
        workers=-1,
        use_multiprocessing=True,)



if __name__ == '__main__':
    train_MTGNN()
    #train_FCN_AD()
    #prepare_solar_data()
    #main()