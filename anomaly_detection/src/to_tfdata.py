import tensorflow as tf
from preprocess import Preprocess


class ToTfData():
    
    def __init__(self) -> None:
        '''
        Attributes:
            batch_size: an integer value indicating the batch size to use
        '''
        self.batch_size = 32

    def train_to_tfdata(self,kpis,labels) -> tf.data.Dataset:
        # Create tf.data.Dataset from train data
        train_ds = tf.data.Dataset.from_tensor_slices((kpis, labels))
        train_ds = train_ds.map(Preprocess().cast)
        train_ds = train_ds.map(Preprocess().one_hot_encode_labels).shuffle(10000).batch(self.batch_size)
        return train_ds


    def val_to_tfdata(self,kpis,labels) -> tf.data.Dataset:
        val_ds = tf.data.Dataset.from_tensor_slices((kpis, labels))
        val_ds = val_ds.map(Preprocess().cast)
        val_ds = val_ds.map(Preprocess().one_hot_encode_labels).batch(self.batch_size)
        return val_ds

    def test_to_tfdata(self,kpis,labels) -> tf.data.Dataset:
        test_ds = tf.data.Dataset.from_tensor_slices((kpis, labels))
        test_ds = test_ds.map(Preprocess().cast)
        test_ds = test_ds.map(Preprocess().one_hot_encode_labels).batch(self.batch_size)
        return test_ds


if __name__ == '__main__':
    pass