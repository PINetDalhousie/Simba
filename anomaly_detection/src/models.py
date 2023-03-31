import tensorflow as tf

class FCN(tf.keras.Model):
    def __init__(self):
        super(FCN, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(8, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(4, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(2)
        
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    pass