import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, class_id, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight(name='true_positives', initializer='zeros')
        self.fp = self.add_weight(name='false_positives', initializer='zeros')
        self.fn = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, self.class_id), tf.equal(y_pred, self.class_id)), tf.float32), axis=None)
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, self.class_id), tf.equal(y_pred, self.class_id)), tf.float32), axis=None)
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, self.class_id), tf.not_equal(y_pred, self.class_id)), tf.float32), axis=None)
        self.tp.assign_add(true_positives)
        self.fp.assign_add(false_positives)
        self.fn.assign_add(false_negatives)

    def result(self): 
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        return f1_score

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


if __name__ == '__main__':
    pass