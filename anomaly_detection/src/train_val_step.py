import tensorflow as tf
from metrics import F1Score
import datetime

class TrainVal:
    
    def __init__(self,model:tf.keras.Model,loss_object,optimizer) -> None:
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer

        # Initialize metrics
        self.initialize_metrics()

        # Initialize tensorboard writer
        self.initialize_tensorboard_writer()

    
    def initialize_tensorboard_writer(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'../logs/{current_time}/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = f'../logs/{current_time}/val'
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)


    def initialize_metrics(self):
        # Initializes train logging metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # define the precision and recall metrics for class 0/non failure
        self.precision_0_train = tf.keras.metrics.Precision(class_id=0)
        self.recall_0_train = tf.keras.metrics.Recall(class_id=0)
        self.f1score_0_train = F1Score()
        # define the precision and recall metrics for class 1/failure
        self.precision_1_train = tf.keras.metrics.Precision(class_id=1)
        self.recall_1_train = tf.keras.metrics.Recall(class_id=1)
        self.f1score_1_train = F1Score()

        # initialize validation logging metrics
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.precision_0_val = tf.keras.metrics.Precision(class_id=0)
        self.recall_0_val = tf.keras.metrics.Recall(class_id=0)
        self.f1score_0_val = F1Score()
        # define the precision and recall metrics for class 1/failure
        self.precision_1_val = tf.keras.metrics.Precision(class_id=1)
        self.recall_1_val = tf.keras.metrics.Recall(class_id=1)
        self.f1score_1_val = F1Score()



    def reset_metrics(self):
        '''
        Reset the metrics at the start of the next epoch
        '''
        # Reset train metrics 
        self.train_loss.reset_states()
        self.precision_0_train.reset_states()
        self.recall_0_train.reset_states()
        self.f1score_0_train.reset_states()
        self.precision_1_train.reset_states()
        self.recall_1_train.reset_states()
        self.f1score_1_train.reset_states()

        # reset validation metrics
        self.val_loss.reset_states()
        self.precision_0_val.reset_states()
        self.recall_0_val.reset_states()
        self.f1score_0_val.reset_states()
        self.precision_1_val.reset_states()
        self.recall_1_val.reset_states()
        self.f1score_1_val.reset_states()


    def update_train_eval_metrics(self,labels,predictions):
        '''
        Updates all logging metrics
        '''
        # update train metrics
        self.precision_0_train.update_state(labels, predictions)
        self.recall_0_train.update_state(labels, predictions)
        self.f1score_0_train.update_state(self.precision_0_train,self.recall_0_train)
        self.precision_1_train.update_state(labels, predictions)
        self.recall_1_train.update_state(labels, predictions)
        self.f1score_1_train.update_state(self.precision_1_train,self.recall_1_train)


    def update_val_eval_metrics(self,labels,predictions):
        # update validation metrics
        self.precision_0_val.update_state(labels, predictions)
        self.recall_0_val.update_state(labels, predictions)
        self.f1score_0_val.update_state(self.precision_0_val,self.recall_0_val)
        self.precision_1_val.update_state(labels, predictions)
        self.recall_1_val.update_state(labels, predictions)
        self.f1score_1_val.update_state(self.precision_1_val,self.recall_1_val)


    def get_train_loss(self):
        return self.train_loss

    def get_train_accuracy(self):
        return self.train_accuracy
    
    def get_val_loss(self):
        return self.val_accuracy

    def get_val_accuracy(self):
        return self.val_accuracy
    
    def write_metrics(self,epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('f1score', self.f1score_1_train.result(),step=epoch)

        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
            tf.summary.scalar('f1score', self.f1score_1_val.result(), step=epoch)
            

    #@tf.function
    def train_step(self,kpis,labels):
        with tf.GradientTape() as tape:
            predictions = self.model(kpis, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        self.train_loss(loss)
        self.update_train_eval_metrics(labels, predictions)


    @tf.function
    def val_step(self,kpis,labels):
        predictions = self.model(kpis, training=False)
        val_loss = self.loss_object(labels, predictions)
        self.val_loss(val_loss)
        self.update_val_eval_metrics(labels, predictions)
    
    





if __name__ == '__main__':
    pass