
from data import KPIData
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models import FCN
from train_val_step import TrainVal, EarlyStopping
from to_tfdata import ToTfData

def main():
    # Gets dataset and performs preprocessing
    data =  KPIData("../data/db_fm_training.txt")
    data.convert_data_for_anomaly_detection()
    kpis, labels = data.seperate_into_kpis_and_label()

    # Split dataset into train and validaiton
    kpis_train, kpis_val, labels_train, labels_val = train_test_split(kpis, labels, test_size=0.2, random_state=42)

    # scale data and save scalar
    kpis_train, kpis_val = KPIData.min_max_scale_train_val(kpis_train,kpis_val)

    # Create tf.data.Dataset for train and validation data
    train_ds = ToTfData().train_to_tfdata(kpis_train,labels_train)
    val_ds = ToTfData().val_to_tfdata(kpis_val,labels_val)
    
    # define the model, loss and optimizer
    model = FCN()
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # define train and validation steps
    train_val = TrainVal(model,loss_object,optimizer)

    # add early stopping
    early_stopping = EarlyStopping(patience=20)

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

        # check for early stopping
        if early_stopping(train_val.val_loss.result()):
            break

    # save model
    model.save("../models/FCN")


if __name__ == '__main__':
    main()