#import  tensorflow as tf

class FullyConnectedNN(tf.keras.Model):
    def __init__(self, output_size):

        super(FullyConnectedNN, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(8, activation='relu')
        self.dense2 = tf.keras.layers.Dense(4, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_size)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    pass