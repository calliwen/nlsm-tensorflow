# -*- coding: utf-8 -*-

import tensorflow as tf
import layer_utils
import match_utils


class SentenceMatchModelGraph(object):
    """
    Create Natural Language Sentence Matching Models.
        -- sentence-sentence pairs
        -- question-answer pairs
    """
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, options=None, global_step=None):
        self.options = options
        self.create_placeholders()
        if options.using_algo == 'bimpm':
            self.create_bimpm_model_graph(num_classes, word_vocab, char_vocab, is_training, global_step=global_step)
        elif options.using_algo == 'mpcnn':
            self.create_mpcnn_model_graph(num_classes, word_vocab, char_vocab, is_training, global_step=global_step)
        elif options.using_algo == 'siameseLSTM':
            self.create_siameseLSTM_model_graph(num_classes, word_vocab, char_vocab, is_training, global_step=global_step)

    def create_placeholders(self):
        self.question_lengths = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.passage_lengths = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.truth = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.in_question_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
        self.in_passage_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        if self.options.with_char:
            self.question_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]

    def create_feed_dict(self, cur_batch, is_training=False):
        feed_dict = {
            self.question_lengths: cur_batch.question_lengths,
            self.passage_lengths: cur_batch.passage_lengths,
            self.in_question_words: cur_batch.in_question_words,
            self.in_passage_words: cur_batch.in_passage_words,
            self.truth : cur_batch.label_truth,
        }

        if self.options.with_char:
            feed_dict[self.question_char_lengths] = cur_batch.question_char_lengths
            feed_dict[self.passage_char_lengths] = cur_batch.passage_char_lengths
            feed_dict[self.in_question_chars] = cur_batch.in_question_chars
            feed_dict[self.in_passage_chars] = cur_batch.in_passage_chars

        return feed_dict


    # ==================================================== BiMPM =================================================
    def create_bimpm_model_graph(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, global_step=None):
        """
        """
        options = self.options
        # ======word representation layer======
        in_question_repres = []
        in_passage_repres = []
        input_dim = 0
        if word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.embedding = tf.placeholder(tf.float32, shape=word_vocab.word_vecs.shape)
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=self.embedding, dtype=tf.float32) # tf.constant(word_vocab.word_vecs)
                
            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim

        if options.with_char and char_vocab is not None:
            char_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_char_vec:
                char_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.char_embedding = tf.get_variable("char_embedding", trainable=char_vec_trainable, 
                                                    initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
            
            input_shape = tf.shape(self.in_question_chars)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = char_vocab.word_dim
            
            in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            quesiton_char_mask = tf.sequence_mask(question_char_lengths, q_char_len, dtype=tf.float32)  # [batch_size*question_len, q_char_len]
            in_question_char_repres = tf.multiply(in_question_char_repres, tf.expand_dims(quesiton_char_mask, axis=-1))

            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            passage_char_mask = tf.sequence_mask(passage_char_lengths, p_char_len, dtype=tf.float32)  # [batch_size*passage_len, p_char_len]
            in_passage_char_repres = tf.multiply(in_passage_char_repres, tf.expand_dims(passage_char_mask, axis=-1))

            (question_char_outputs_fw, question_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_question_char_repres, options.char_lstm_dim,
                    input_lengths=question_char_lengths,scope_name="char_lstm", reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            question_char_outputs_fw = layer_utils.collect_final_step_of_lstm(question_char_outputs_fw, question_char_lengths - 1)
            question_char_outputs_bw = question_char_outputs_bw[:, 0, :]
            question_char_outputs = tf.concat(axis=1, values=[question_char_outputs_fw, question_char_outputs_bw])
            question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, 2*options.char_lstm_dim])  # [batch_size, question_len, 2*options.char_lstm_dim]

            (passage_char_outputs_fw, passage_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_passage_char_repres, options.char_lstm_dim,
                    input_lengths=passage_char_lengths, scope_name="char_lstm", reuse=True,
                    is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            passage_char_outputs_fw = layer_utils.collect_final_step_of_lstm(passage_char_outputs_fw, passage_char_lengths - 1)
            passage_char_outputs_bw = passage_char_outputs_bw[:, 0, :]
            passage_char_outputs = tf.concat(axis=1, values=[passage_char_outputs_fw, passage_char_outputs_bw])
            passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, 2*options.char_lstm_dim])  # [batch_size, passage_len, 2*options.char_lstm_dim]
            
            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)

            input_dim += 2*options.char_lstm_dim

        in_question_repres = tf.concat(axis=2, values=in_question_repres) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2, values=in_passage_repres) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))

        mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)

        # in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
        # in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(mask, axis=-1))

        # ========Bilateral Matching=====
        (match_representation, match_dim) = match_utils.bilateral_match_func(in_question_repres, in_passage_repres,
                        self.question_lengths, self.passage_lengths, question_mask, mask, input_dim, is_training, options=options)

        #========Prediction Layer=========
        # match_dim = 4 * self.options.aggregation_lstm_dim
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim/2, num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.nn.relu(logits)
        if is_training: logits = tf.nn.dropout(logits, (1 - options.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1
        
        self.prob = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.prob, 1)
        
        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        
        if not is_training: return
        
        # if options.with_f1_metric:
        #     # acc, acc_op = tf.metrics.accuracy(labels=self.truth, predictions=self.predictions)
        #     precision, pre_op = tf.metrics.precision(labels=self.truth, predictions=self.predictions)
        #     recall, rec_op = tf.metrics.recall(labels=self.truth, predictions=self.predictions)
        #     f1 = 2 * precision * recall / (precision + recall + 1e-6)
        #     self.loss = self.loss - 0.1 * tf.reduce_mean(f1)
        
        tvars = tf.trainable_variables()
        if self.options.lambda_l1>0.0:
            l1_loss = tf.add_n([tf.contrib.layers.l1_regularizer(self.options.lambda_l1)(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + l1_loss
        if self.options.lambda_l2>0.0:
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            # l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
            l2_loss = tf.add_n([tf.contrib.layers.l2_regularizer(self.options.lambda_l2)(v) for v in tvars])
            self.loss = self.loss + l2_loss

        if self.options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)

        grads = layer_utils.compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)


    # ==================================================== MPCNN =================================================
    def create_mpcnn_model_graph(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, global_step=None):
        """
        """
        options = self.options
        # ======word representation layer======
        in_question_repres = []
        in_passage_repres = []
        input_dim = 0
        if word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.embedding = tf.placeholder(tf.float32, shape=word_vocab.word_vecs.shape)
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=self.embedding, dtype=tf.float32) # tf.constant(word_vocab.word_vecs)

            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim

        in_question_repres = tf.concat(axis=2, values=in_question_repres) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2, values=in_passage_repres) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))

        mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)
        
        in_question_repres = tf.expand_dims(in_question_repres, -1) # [batch_size, question_len, word_dim, 1]
        in_passage_repres = tf.expand_dims(in_passage_repres, -1) # [batch_size, passage_len, word_dim, 1]

        # ======Multi-perspective CNN Matching======
        filter_sizes = options.filter_sizes
        num_filters = options.num_filters
        poolings = list([tf.reduce_max, tf.reduce_min, tf.reduce_mean])[:options.num_poolings]
        
        W1 = [tf.get_variable("W1_%s" %i, initializer=tf.truncated_normal([filter_sizes[i], input_dim, 1, num_filters[0]], stddev=0.1), dtype=tf.float32) for i in range(len(filter_sizes))]
        b1 = [tf.get_variable("b1_%s" %i, initializer=tf.constant(0.01, shape=[num_filters[0]]), dtype=tf.float32) for i in range(len(filter_sizes))]

        W2 = [tf.get_variable("W2_%s" %i, initializer=tf.truncated_normal([filter_sizes[i], input_dim, 1, num_filters[1]], stddev=0.1), dtype=tf.float32) for i in range(len(filter_sizes)-1)]
        b2 = [tf.get_variable("b2_%s" %i, initializer=tf.constant(0.01, shape=[num_filters[1], input_dim]), dtype=tf.float32) for i in range(len(filter_sizes)-1)]
        
        sent1_blockA = layer_utils.build_block_A(in_question_repres, filter_sizes, poolings, W1, b1, is_training) # len(poolings) * len(filter_sizes) * [batch_size， 1， num_filters_A]
        sent2_blockA = layer_utils.build_block_A(in_passage_repres, filter_sizes, poolings, W1, b1, is_training) # len(poolings) * len(filter_sizes) * [batch_size， 1， num_filters_A]

        sent1_blockB = layer_utils.build_block_B(in_question_repres, filter_sizes, poolings, W2, b2, is_training) # (len(poolings))-1 * (len(filter_sizes)-1) * [batch_size， embed_size， num_filters_B]
        sent2_blockB = layer_utils.build_block_B(in_passage_repres, filter_sizes, poolings, W2, b2, is_training) # (len(poolings))-1 * (len(filter_sizes)-1) * [batch_size， embed_size， num_filters_B]

        (match_representation, match_dim) = match_utils.mpcnn_match_func(sent1_blockA, sent2_blockA, 
                                                    sent1_blockB, sent2_blockB, poolings, filter_sizes, num_filters)
    
        #========Prediction Layer=========
        w_0 = tf.get_variable("w_0", [match_dim, int(match_dim/2)], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [int(match_dim/2)], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [int(match_dim/2), num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.nn.relu(logits)
        if is_training: logits = tf.nn.dropout(logits, (1 - options.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1
        
        self.prob = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.prob, 1)
        
        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))
        
        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        if not is_training: return
        
        if options.with_f1_metric:
            # acc, acc_op = tf.metrics.accuracy(labels=self.truth, predictions=self.predictions)
            precision, pre_op = tf.metrics.precision(labels=self.truth, predictions=self.predictions)
            recall, rec_op = tf.metrics.recall(labels=self.truth, predictions=self.predictions)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            self.loss = self.loss - 0.1 * tf.reduce_mean(f1)
        
        tvars = tf.trainable_variables()
        if self.options.lambda_l1>0.0:
            l1_loss = tf.add_n([tf.contrib.layers.l1_regularizer(self.options.lambda_l1)(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + l1_loss
        if self.options.lambda_l2>0.0:
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            l2_loss = tf.add_n([tf.contrib.layers.l2_regularizer(self.options.lambda_l2)(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + l2_loss

        if self.options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)

        grads = layer_utils.compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)


    # ==================================================== SiameseLSTM ===========================================
    def create_siameseLSTM_model_graph(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, global_step=None):
        """
        """
        options = self.options
        # ======word representation layer======
        in_question_repres = []
        in_passage_repres = []
        input_dim = 0
        if word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.embedding = tf.placeholder(tf.float32, shape=word_vocab.word_vecs.shape)
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=self.embedding, dtype=tf.float32) # tf.constant(word_vocab.word_vecs)
                
            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim

        in_question_repres = tf.concat(axis=2, values=in_question_repres) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2, values=in_passage_repres) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))

        passage_mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)

        # ======BiLSTM context layer======
        for i in range(options.context_layer_num): # support multiple context layer
            with tf.variable_scope('bilstm-layer-{}'.format(i)):
                # contextual lstm for both passage and question
                in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
                (question_context_representation_fw, question_context_representation_bw,
                 in_question_repres) = layer_utils.my_lstm_layer(
                        in_question_repres, options.context_lstm_dim, input_lengths=self.question_lengths,scope_name="context_represent",
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
                
                # Encode the second sentence, using the same LSTM weights.
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(passage_mask, axis=-1))
                (passage_context_representation_fw, passage_context_representation_bw, 
                 in_passage_repres) = layer_utils.my_lstm_layer(
                        in_passage_repres, options.context_lstm_dim, input_lengths=self.passage_lengths, scope_name="context_represent",
                        reuse=True, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)

        if options.lstm_out_type == 'mean':
            question_context_representation_fw = layer_utils.collect_mean_step_of_lstm(question_context_representation_fw)
            question_context_representation_bw = layer_utils.collect_mean_step_of_lstm(question_context_representation_bw)
            passage_context_representation_fw = layer_utils.collect_mean_step_of_lstm(passage_context_representation_fw)
            passage_context_representation_bw = layer_utils.collect_mean_step_of_lstm(passage_context_representation_bw)
        elif options.lstm_out_type == 'end':
            question_context_representation_fw = layer_utils.collect_final_step_of_lstm(question_context_representation_fw, self.question_lengths - 1)
            question_context_representation_bw = question_context_representation_bw[:, 0, :]
            passage_context_representation_fw = layer_utils.collect_final_step_of_lstm(passage_context_representation_fw, self.passage_lengths - 1)
            passage_context_representation_bw = passage_context_representation_bw[:, 0, :]
        
        question_context_outputs = tf.concat(axis=1, values=[question_context_representation_fw, question_context_representation_bw])
        passage_context_outputs = tf.concat(axis=1, values=[passage_context_representation_fw, passage_context_representation_bw])
        
        (match_representation, match_dim) = match_utils.siameseLSTM_match_func(question_context_outputs, passage_context_outputs, options.context_lstm_dim)

        #========Prediction Layer=========
        w_0 = tf.get_variable("w_0", [match_dim, int(match_dim/2)], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [int(match_dim/2)], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [int(match_dim/2), num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.nn.relu(logits)
        if is_training: logits = tf.nn.dropout(logits, (1 - options.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1
        
        self.prob = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.prob, 1)
        
        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))
        
        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        if not is_training: return
        
        tvars = tf.trainable_variables()
        if self.options.lambda_l1>0.0:
            l1_loss = tf.add_n([tf.contrib.layers.l1_regularizer(self.options.lambda_l1)(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + l1_loss
        if self.options.lambda_l2>0.0:
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            l2_loss = tf.add_n([tf.contrib.layers.l2_regularizer(self.options.lambda_l2)(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + l2_loss

        if self.options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)

        grads = layer_utils.compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)

