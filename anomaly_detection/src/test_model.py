
from data import KPIData
from to_tfdata import ToTfData
import tensorflow as tf
from metrics import F1Score
from test_metrics import TestMetrics


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