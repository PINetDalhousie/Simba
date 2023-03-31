

import pandas as pd
from data import KPIData
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models import FCN
from train_val_step import TrainVal
from preprocess import Preprocess

def main():
    
    # Get dataset
    data = pd.read_csv("../data/db_fm_training.txt",sep='\s+',comment='%',
                             names=['Retainability', 'HOSR', 'RSRP', 'RSRQ', 'SINR', 'Throughput', 'Distance', 'FaultCause'])
    
    # Convert data into anomaly or normal dataset
    data['FaultCause'] = data['FaultCause'].apply(lambda x: 0.0 if x==7.0 else 1.0)

    # Seperate data into kpis and label
    kpis = data.drop(["FaultCause"],axis=1)
    labels = data['FaultCause']

    # Split dataset into train and validaiton
    kpis_train, kpis_val, labels_train, labels_val = train_test_split(kpis, labels, test_size=0.2, random_state=42)

    # normalize data
    min_max_scaler = MinMaxScaler()
    kpis_train = min_max_scaler.fit_transform(kpis_train)
    kpis_val = min_max_scaler.transform(kpis_val)

    # Create tf.data.Dataset object from dataset
    train_ds = tf.data.Dataset.from_tensor_slices((kpis_train, labels_train))
    train_ds = train_ds.map(Preprocess().cast)
    train_ds = train_ds.map(Preprocess().one_hot_encode_labels).shuffle(10000).batch(32)
    
    val_ds = tf.data.Dataset.from_tensor_slices((kpis_val, labels_val))
    val_ds = val_ds.map(Preprocess().cast)
    val_ds = val_ds.map(Preprocess().one_hot_encode_labels).batch(32)

    # define the model
    model = FCN()

    # Choose an optimizer and loss function for training
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # define train and validation steps
    train_val = TrainVal(model,loss_object,optimizer)

    # train the model
    EPOCHS = 1000

    for epoch in range(EPOCHS):
        # reset metrics
        train_val.reset_metrics()

        for kpis, labels in train_ds:
            train_val.train_step(kpis, labels)

        for kpis, labels in val_ds:
            train_val.val_step(kpis, labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Train Loss: {train_val.train_loss.result()}, '
            f'Train F1score 1: {train_val.f1score_1_train.result() * 100}, ',
            f'Val Loss: {train_val.val_loss.result()}, '
            f'Val F1score 1: {train_val.f1score_1_val.result() * 100}, '
        )

        # Write metrics to tensorboard
        train_val.write_metrics(epoch)


if __name__ == '__main__':
    main()