import tensorflow as tf
from metrics import F1Score

class TestMetrics:
    def __init__(self):
        # define the precision and recall metrics for class 0/non failure
        self.precision_0_test = tf.keras.metrics.Precision(class_id=0)
        self.recall_0_test = tf.keras.metrics.Recall(class_id=0)
        self.f1score_0_test = F1Score()
        # define the precision and recall metrics for class 1/failure
        self.precision_1_test = tf.keras.metrics.Precision(class_id=1)
        self.recall_1_test = tf.keras.metrics.Recall(class_id=1)
        self.f1score_1_test = F1Score()

    def update_test_metrics(self,labels,predictions):
        self.precision_0_test.update_state(labels, predictions)
        self.recall_0_test.update_state(labels, predictions)
        self.f1score_0_test.update_state(self.precision_0_test,self.recall_0_test)
        self.precision_1_test.update_state(labels, predictions)
        self.recall_1_test.update_state(labels, predictions)
        self.f1score_1_test.update_state(self.precision_1_test,self.recall_1_test)

if __name__ == '__main__':
    pass