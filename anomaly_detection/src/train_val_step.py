import tensorflow as tf

class TrainVal:
    
    def __init__(self,model:tf.keras.Model,loss_object,optimizer) -> None:
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer

        # Initializes logging metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def get_train_loss(self):
        return self.train_loss

    def get_train_accuracy(self):
        return self.train_accuracy
    
    def get_test_loss(self):
        return self.test_accuracy

    def get_test_accuracy(self):
        return self.test_accuracy
    

    #@tf.function
    def train_step(self,kpis,labels):
        with tf.GradientTape() as tape:
            predictions = self.model(kpis, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    #@tf.function
    def test_step(self,kpis,labels):
        predictions = self.model(kpis, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
    
    





if __name__ == '__main__':
    pass