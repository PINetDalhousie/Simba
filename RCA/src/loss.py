import tensorflow as tf
import pandas as pd

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Custom loss class for computing weighted categorical cross-entropy loss.
    
    This class inherits from `tf.keras.losses.Loss` and overrides the `call` method
    to compute the weighted categorical cross-entropy loss.
    """

    def __init__(self, class1_ratio:float, name='weighted_categorical_crossentropy'):
        """Initializes a new instance of the WeightedCategoricalCrossentropy class.
        
        Args:
            ratio: A float representing the weight for the minority class.
            name: A string representing the name of the loss. Defaults to 'weighted_categorical_crossentropy'.
        """
        super().__init__(name=name)
        self.min_maj_ratio = class1_ratio
        self.class_weight = tf.constant([[class1_ratio, 1.0 - class1_ratio]],dtype=tf.float32)
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


    def call(self, y_true, y_pred):
        """Computes the weighted categorical cross-entropy loss.
        
        Args:
            y_true: A tensor of shape (batch_size, num_classes) representing the true labels.
            y_pred: A tensor of shape (batch_size, num_classes) representing the predicted probabilities.
            
        Returns:
            A scalar tensor representing the weighted categorical cross-entropy loss.
        """
        weights = tf.matmul(y_true, tf.transpose(self.class_weight))
        sample_losses = tf.reshape(self.categorical_crossentropy(y_true, y_pred),[-1,1])
        loss = tf.matmul(tf.transpose(sample_losses),weights)
        return loss
    
    

if __name__ == "__main__":
    pass