
from data import KPIData
from to_tfdata import ToTfData
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


def main():
    # Gets dataset and performs preprocessing
    data =  KPIData("../data/db_fm_training.txt")
    data.convert_data_for_anomaly_detection()
    kpis, labels = data.seperate_into_kpis_and_label()
    kpis = KPIData.scale_test("../scalers/min_max_scaler.save",kpis)
    test_ds = ToTfData().test_to_tfdata(kpis,labels)

    # load the model
    model = tf.keras.models.load_model("../models/FCN")
    
    # initialize test metrics
    test_metrics = TestMetrics()

    # perform inference on model
    for kpis, labels in test_ds:
        predictions = model(kpis, training=False)
        test_metrics.update_test_metrics(labels,predictions)

    # print metric results
    print(
        f'Test Precision 0: {test_metrics.precision_0_test.result() * 100}, ',
        f'Test Recall 0: {test_metrics.recall_0_test.result() * 100}, ',
        f'Test F1score 0: {test_metrics.f1score_0_test.result() * 100}, ',
        f'Test Precision 1: {test_metrics.precision_1_test.result() * 100}, ',
        f'Test Recall 1: {test_metrics.recall_1_test.result() * 100}, ',
        f'Test F1score 1: {test_metrics.f1score_1_test.result() * 100}, ',
        )


if __name__ == '__main__':
    main()