import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1_score', initializer='zeros')

    def update_state(self,precision:tf.keras.metrics.Precision,recall:tf.keras.metrics.Recall):
        precision = precision.result()
        recall = recall.result()
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-6))
        self.f1_score.assign(f1_score)

    def result(self):
        return self.f1_score

    def reset_states(self):
        self.f1_score.assign(0.0)



if __name__ == '__main__':
    pass