

import pandas as pd
from data import KPIData
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models import FCN
from train_val_step import TrainVal

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
    kpis_train, kpis_test, labels_train, labels_test = train_test_split(kpis, labels, test_size=0.2, random_state=42)

    # normalize data
    min_max_scaler = MinMaxScaler()
    kpis_train = min_max_scaler.fit_transform(kpis_train)
    kpis_test = min_max_scaler.transform(kpis_test)

    # Create tf.data.Dataset object from dataset
    train_ds = tf.data.Dataset.from_tensor_slices((kpis_train, labels_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((kpis_test, labels_test)).batch(32)

    # define the model
    model = FCN()

    # Choose an optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # define train and validation steps
    train_val = TrainVal(model,loss_object,optimizer)

    # train the model
    EPOCHS = 1000

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_val.train_loss.reset_states()
        train_val.train_accuracy.reset_states()
        train_val.test_loss.reset_states()
        train_val.test_accuracy.reset_states()

        for kpis, labels in train_ds:
            train_val.train_step(kpis, labels)

        for kpis, labels in test_ds:
            train_val.test_step(kpis, labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_val.train_loss.result()}, '
            f'Accuracy: {train_val.train_accuracy.result() * 100}, '
            f'Test Loss: {train_val.test_loss.result()}, '
            f'Test Accuracy: {train_val.test_accuracy.result() * 100}'
        )


if __name__ == '__main__':
    main()