import tensorflow as tf

class Preprocess:
    
    def __init__(self) -> None:
        '''
        Attributes:
            depth: an integer to indicate number of classes in labels
        '''
        self.depth = 2


    def cast(self,kpis,labels):
        return tf.cast(kpis,dtype=tf.float32),tf.cast(labels,dtype=tf.int32)


    def one_hot_encode_labels(self,kpis,labels):
        '''
        This function encodes labels into one hot encoded tensors
        '''
        return kpis, tf.one_hot(labels, depth=2)


if __name__ == '__main__':
    pass