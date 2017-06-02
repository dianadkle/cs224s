from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


class Config:
    random_seed = 101

    seq_length = 50 ########
    embed_size = 13 ########
    n_classes = 5  
    state_size = seq_length
 
    n_epochs = 10
    batch_size = 10
    early_term_criterion = 1.0e-6
    n_epochs_per_ckpt = 5

    dtype = tf.float64
    decay_rate = 0.95 # decay per epoch
    max_grad_norm = 10.0

    # hyperparameters
    starter_lr = 0.001
    dropout = 0.20
    l2_reg = 0.001


class LstmNeuralNetwork(object):

    def __init__(self, train_dir="data/train", seed=101):
        # set random seed
        random.seed(Config.random_seed)
        np.random.seed(Config.random_seed)
        tf.set_random_seed(Config.random_seed)

        # create neural network
        self.model = NeuralNetworkModel(bilstm=True)

        # save your model parameters/checkpoints here
        self.train_dir = train_dir
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # start tf session
        self.session = tf.Session()

        # initialize model
        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            print("read parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("created model with fresh parameters")
            self.session.run(tf.global_variables_initializer())
        print("n_params: %d" % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

    def close():
        if self.session is not None:
            self.session.close()
            self.session = None

    def fit(self, X, y):
        assert y.shape == (X.shape[0],)
        n_samples = y.shape[0]

        # initialize exponential decay_rate
        self.model.global_step = 0
        self.model.decay_steps = Config.n_epochs

        # calculate checkpoint frequencies
        n_batches = int(n_samples / Config.batch_size)

        # split in minibatch of size batch_size
        for epoch in xrange(Config.n_epochs):
            total_loss, total_grad_norm, batch_count = 0.0, 0.0, 0

            for start_idx in xrange(0, n_samples, Config.batch_size):
                end_idx = min(start_idx + Config.batch_size, n_samples)

                # prepare input data and label data
                input_feed = self.model.set_input_feed(X[start_idx:end_idx], y[start_idx:end_idx], Config.dropout)
                output_feed = [self.model.train_op, self.model.loss, self.model.grad_norm]

                # train this batch
                _, loss, grad_norm = self.session.run(output_feed, input_feed)

                # update cumulative stats
                total_loss += loss
                total_grad_norm += grad_norm
                batch_count += 1

            # increment global step
            self.model.global_step += 1

            # checkpoint the model for each epoch
            if (epoch+1) % Config.n_epochs_per_ckpt == 0:
                save_path = self.model.saver.save(self.session, "%s/model_epoch_%d.ckpt" % (self.train_dir, epoch))
                
            # compute epoch loss
            epoch_loss = total_loss / float(batch_count)
            epoch_grad_norm = total_grad_norm / float(batch_count)
            print("epoch = %d, loss = %6.4f, grad_norm = %6.4f" % (epoch, epoch_loss, epoch_grad_norm))

            # check for early termination
            if epoch_grad_norm < Config.early_term_criterion:
                print("EARLY TERMINATION")
                return
        print("Training is done")

    def predict(self, X):
        n_samples = X.shape[0]
        y_output = np.zeros(n_samples, dtype=np.int)

        batch_size = Config.batch_size
        for start_idx in xrange(0, n_samples, batch_size):
            # calculate end_idx
            end_idx = min(start_idx + batch_size, n_samples)
            
            # prepare input data and label data
            input_feed = self.model.set_input_feed(X[start_idx:end_idx])
            output_feed = [self.model.evals, self.model.softmax]

            # run returns a numpy ndarray
            evals, _ = self.session.run(output_feed, input_feed) # (batch_size)
            y_output[start_idx:end_idx] = evals
        
        self.X = X
        self.y = y_output
        return y_output

    def score(self, X, y):
        assert y.shape == (X.shape[0],)
        n_samples = y.shape[0]
        if self.X is not X:
            self.predict(X)
        
        accuracy = np.sum(self.y == y) / float(n_samples)
        return accuracy


class NeuralNetworkModel(object):
    
    def __init__(self, name="NeuralNework", bilstm=True):
        self.name = name
        if bilstm:
            self.encoder = Encoder2(self.name)
            self.decoder = Decoder2(self.name)
        else:
            self.encoder = Encoder(self.name)
            self.decoder = Decoder(self.name)

        # placeholders
        self.input_placeholder = tf.placeholder(Config.dtype, shape=(None, Config.seq_length * Config.embed_size))
        self.input_seq_length_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.dropout_placeholder = tf.placeholder(Config.dtype)

        # graph
        with tf.variable_scope(self.name, initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        self.saver = tf.train.Saver()

    def setup_embeddings(self):
        self.x = tf.reshape(self.input_placeholder, [-1, Config.seq_length, Config.embed_size]) # (batch_size, seq_length, embed_size)
        self.seq_length = self.input_seq_length_placeholder

    def setup_system(self):
        # connect components together
        encoded = self.encoder.encode(self.x, self.seq_length)
        self.preds = self.decoder.decode(encoded, self.seq_length, self.dropout_placeholder) # (batch_size, n_classes)

        # for evaluation
        self.softmax = tf.nn.softmax(self.preds) # (batch_size, n_classes)
        self.evals = tf.argmax(self.softmax, 1) # (batch_size)

    def setup_loss(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.preds, self.labels_placeholder)
        self.loss = tf.reduce_mean(losses)
        l2_cost = 0.0
        for var in tf.trainable_variables():
            if len(var.get_shape()) > 1:
                l2_cost += tf.nn.l2_loss(var)
        self.loss += Config.l2_reg * l2_cost

        # lr = starter_lr * decay_rate ^ (global_step / decay_steps)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.decay_steps = tf.Variable(100, trainable=False, dtype=tf.int32) # temporary value
        self.lr = tf.train.exponential_decay(Config.starter_lr,
                            self.global_step, self.decay_steps,
                            Config.decay_rate, staircase=True)

        # create optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        # gradient clipping
        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grads = [x[0] for x in grads_and_vars]
        if Config.max_grad_norm > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, Config.max_grad_norm)
        self.grad_norm = tf.global_norm(grads)
        grads_and_vars = [(grads[i], x[1]) for i, x in enumerate(grads_and_vars)]
        self.train_op = optimizer.apply_gradients(grads_and_vars)
   
    def set_input_feed(self, X_batch, y_batch=None, dropout=0.0):
        seq_length = np.array([Config.seq_length for _ in xrange(X_batch.shape[0])])
        input_feed = {}
        input_feed[self.input_placeholder] = X_batch
        input_feed[self.input_seq_length_placeholder] = seq_length
        input_feed[self.dropout_placeholder] = 1.0 - dropout
        if y_batch is not None:
            input_feed[self.labels_placeholder] = y_batch
        return input_feed



class Encoder2(object):

    def __init__(self, name="algo"):
        self.name = name + ".Encoder2"
        self.cell = Lstm2(Config.state_size, name=self.name)

    def encode(self, x, seq_length):
        output_fw, output_bw, state_fw, state_bw = self.cell.run(x, seq_length)
        encoded = tf.concat_v2([output_fw, output_bw], 2) # (batch_size, seq_length, 2 * state_size)
        return encoded

    def regularization(self):
        return 0


class Decoder2(object):

    def __init__(self, name):
        self.name = name + ".Decoder2"
        self.cell = Lstm2(Config.state_size, name=self.name)

        with vs.variable_scope(self.name):
            self.W = tf.get_variable("affine.weight", shape=(Config.seq_length * 2 * Config.state_size, Config.n_classes),
                                        dtype=Config.dtype, initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable("affine.bias", shape=(Config.n_classes),
                                        dtype=Config.dtype, initializer=tf.constant_initializer(0))

    def decode(self, encoded, seq_length, dropout=1.0):
        output_fw, output_bw, state_fw, state_bw = self.cell.run(encoded, seq_length)
        x = tf.concat_v2([output_fw, output_bw], 2) # (batch_size, seq_length, 2 * state_size)
        x = tf.reshape(x, [-1, Config.seq_length * 2 * Config.state_size])

        # output layer
        out_drop = tf.nn.dropout(x, dropout) # (batch_size, seq_length * 2 * state_size)
        preds = tf.matmul(out_drop, self.W) + self.b # (batch_size, n_classes)        
        return preds

    def regularization(self):
        return tf.nn.l2_loss(self.W)



class Lstm2(object): # bidirectional LSTM

    def __init__(self, n_units, name="algo"):
        self.name = name + ".Lstm2"
        with vs.variable_scope(self.name):
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(n_units, state_is_tuple=True)
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(n_units, state_is_tuple=True)

    def run(self, x, seq_length, init_state_fw=None, init_state_bw=None):
        with vs.variable_scope(self.name):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                                    self.cell_fw, self.cell_bw, x, dtype=Config.dtype,
                                    sequence_length=seq_length,
                                    initial_state_fw=init_state_fw,
                                    initial_state_bw=init_state_bw)

        output_fw, output_bw = outputs
        state_fw, state_bw = states
        return (output_fw, output_bw, state_fw, state_bw)


