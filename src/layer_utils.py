# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import nn_ops
import tensorflow.contrib.slim as slim

def my_lstm_layer(input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                  dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        if use_cudnn:
            inputs = tf.transpose(input_reps, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                    name="{}_cudnn_bi_lstm".format(scope_name), dropout=dropout_rate if is_training else 0)
            outputs, _ = lstm(inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            f_rep = outputs[:, :, 0:lstm_dim]
            b_rep = outputs[:, :, lstm_dim:2*lstm_dim]
        else:
            context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            if is_training:
                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
            context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return (f_rep, b_rep, outputs)

def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr

def cosine_distance(y1, y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm

def euclidean_distance(y1, y2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance

def l1_distance(y1, y2):
    distance = tf.reduce_sum(tf.abs(tf.subtract(y1, y2)), axis=-1)
    return distance

def cross_entropy(logits, truth, mask=None):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]
    if mask is not None: logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
    result = tf.multiply(truth, log_predictions) # [batch_size, passage_len]
    if mask is not None: result = tf.multiply(result, mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]

def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    # feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    with tf.variable_scope(scope or "projection_layer"):
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs # [batch_size, passage_len, output_size]

def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    # feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, activation_func=tf.tanh, scope_name=None, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        for i in range(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = highway_layer(in_val, output_size,activation_func=activation_func, scope=cur_scope_name)
    return in_val

def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)

def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result # [batch_size, dim]

def collect_mean_step_of_lstm(lstm_representation, lengths=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    lstm_representation: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.
    lengths: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    # shape (batch_size, lengths)
    lstm_representation_sum = tf.reduce_sum(lstm_representation, axis=-2)
    if lengths is None:
        lengths = tf.shape(lstm_representation)[-2]
    # Expand sequence length from shape (batch_size,) to
    # (batch_size, 1) for broadcasting to work.
    expanded_sequence_length = tf.cast(tf.expand_dims(lengths, -1),
                                        "float32") + 1e-08
    # Now, divide by the length of each sequence.
    # shape (batch_size, lengths)
    mean_pooled_input = lstm_representation_sum / expanded_sequence_length
    return mean_pooled_input # [batch_size, dim]

def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size]) # [batch_size, pair_size]

    indices = tf.stack((batch_nums, positions), axis=2) # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs


def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value

def weighted_sum(atten_scores, in_values):
    '''

    :param atten_scores: # [batch_size, len1, len2]
    :param in_values: [batch_size, len2, dim]
    :return:
    '''
    return tf.matmul(atten_scores, in_values)

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]




def build_block_A(x, filter_sizes, poolings, W1, b1, is_training):
    out = []
    with tf.name_scope("bulid_block_A"):
        for pooling in poolings:
            pools = []
            for i, ws in enumerate(filter_sizes):
                with tf.name_scope("conv-pool-%s" % ws):
                    # Convolution Layer
                    conv = tf.nn.conv2d(x, W1[i], strides=[1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=True, name='conv')
                    # Apply nonlinearity
                    conv = tf.nn.relu(tf.nn.bias_add(conv, b1[i]), name="relu")  # [batch_size, sentence_length-ws+1, 1, num_filters_A]
                    # Add batch normalization
                    # conv = tf.layers.batch_normalization(conv, axis=0)
                    # conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=is_training)
                    # Pooling over the outputs
                    pool = pooling(conv, axis=1)  # [batch_size, 1, num_filters_A]
                pools.append(pool)
            out.append(pools)
        return out  # len(poolings) * len(filter_sizes) * [batch_size， 1， num_filters_A]

def build_block_B(x, filter_sizes, poolings, W2, b2, is_training):
    out = []
    with tf.name_scope("bulid_block_B"):
        for pooling in poolings[:-1]:
            pools = []
            for i, ws in enumerate(filter_sizes[:-1]):
                with tf.name_scope("per_conv-pool-%s" % ws):
                    pool = per_dim_conv_layer(x, W2[i], b2[i], pooling, is_training)
                pools.append(pool)
            out.append(pools)
        return out  # len(poolings)-1 * len(filter_sizes)-1 * [batch_size， embed_size， num_filters_B]

def per_dim_conv_layer(x, w, b, pooling, is_training):
    '''
    per_dim convolution
    :param x: [batch_size, sentence_length, embed_size, 1]
    :param w: [filter_size, embed_size, 1, num_filters]
    :param b: [num_filters, embed_size]
    :param pooling:
    :return:
    '''
    input_unstack = tf.unstack(x, axis=2)
    w_unstack = tf.unstack(w, axis=1)
    b_unstack = tf.unstack(b, axis=1)
    convs = []
    for i in range(x.get_shape()[2]):
        conv = tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID", use_cudnn_on_gpu=True, name='per_conv')
        conv = tf.nn.relu(tf.nn.bias_add(conv, b_unstack[i]), name="relu")
        # conv = tf.layers.batch_normalization(conv, axis=0)
        # conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=is_training)
        convs.append(conv)
    conv = tf.stack(convs, axis=2)  # [batch_size, sentence_length-ws+1, embed_size, num_filters_B]
    pool = pooling(conv, axis=1)  # [batch_size, embed_size, num_filters_B]
    return pool





def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
        for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article
        
        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
