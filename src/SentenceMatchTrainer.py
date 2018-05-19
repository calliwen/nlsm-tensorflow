# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import json

from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils

def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt')
    for line in infile:
        line = line.strip()#.decode('utf-8')
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = re.split("\\s+",items[1].lower())
        sentence2 = re.split("\\s+",items[2].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def output_probs(probs, label_vocab):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()

def evaluation(sess, valid_graph, devDataStream, options, outpath=None, label_vocab=None):
    if outpath is not None:
        result_json = {}
    total = 0
    correct = 0
    if options.with_f1_metric: tp = fp = fn = tn = 0
    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=False)
        [cur_correct, probs, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        correct += cur_correct
        if options.with_f1_metric:
            for i in range(cur_batch.batch_size):
                (label, sentence1, sentence2, _, _, _, _, _, cur_ID) = cur_batch.instances[i]
                if int(label) == 1 and int(label_vocab.getWord(predictions[i])) == 1: tp += 1
                if int(label) == 0 and int(label_vocab.getWord(predictions[i])) == 1: fp += 1
                if int(label) == 1 and int(label_vocab.getWord(predictions[i])) == 0: fn += 1
                if int(label) == 0 and int(label_vocab.getWord(predictions[i])) == 0: tn += 1
        if outpath is not None:
            for i in range(cur_batch.batch_size):
                (label, sentence1, sentence2, _, _, _, _, _, cur_ID) = cur_batch.instances[i]
                result_json[cur_ID] = {
                    "ID": cur_ID,
                    "truth": label,
                    "sent1": sentence1,
                    "sent2": sentence2,
                    "prediction": label_vocab.getWord(predictions[i]),
                    "probs": output_probs(probs[i], label_vocab),
                }
    if outpath is not None:
        with open(outpath, 'w') as outfile:
            json.dump(result_json, outfile)
    accuracy = correct / float(total)
    if options.with_f1_metric:
        eps = 1e-6
        precision = tp / float(tp + fp + eps)
        recall = tp / float(tp + fn + eps)
        f1_score = 2 * precision * recall / float(precision + recall + eps)
        return accuracy, f1_score, precision, recall
    return accuracy

def train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, options, best_path, label_vocab):
    best_metric = -1
    for epoch in range(options.max_epochs):
        print('Train in epoch %d' % epoch)
        # training
        trainDataStream.shuffle()
        num_batch = trainDataStream.get_num_batch()
        start_time = time.time()
        total_loss = 0
        for batch_index in range(num_batch):  # for each batch
            cur_batch = trainDataStream.get_batch(batch_index)
            feed_dict = train_graph.create_feed_dict(cur_batch, is_training=True)
            _, loss_value = sess.run([train_graph.train_op, train_graph.loss], feed_dict=feed_dict)
            total_loss += loss_value
            if batch_index % 100 == 0:
                print('{} '.format(batch_index), end="")
                sys.stdout.flush()
        print()
        duration = time.time() - start_time
        print('Epoch %d: loss = %.5f (%.3f sec)' % (epoch, total_loss / num_batch, duration))
        # evaluation
        start_time = time.time()
        if options.with_f1_metric:
            acc, f1, precision, recall = evaluation(sess, valid_graph, devDataStream, options, outpath=options.out_path, label_vocab=label_vocab)
            metric = acc #0.1*acc + 0.9*f1
            duration = time.time() - start_time
            print("Accuracy: %.5f" % acc)
            print("F1-score: %.5f" % f1)
            print("Metric: %.5f" % metric)
            print("Precision: %.5f" % precision)
            print("Recall: %.5f" % recall)
            print('Evaluation time: %.3f sec' % (duration))
        else:
            acc = evaluation(sess, valid_graph, devDataStream, options, outpath=options.out_path, label_vocab=label_vocab)
            metric = acc
            duration = time.time() - start_time
            print("Accuracy: %.5f" % acc)
            print("Metric: %.5f" % metric)
            print('Evaluation time: %.3f sec' % (duration))
        if metric >= best_metric:
            best_metric = metric
            saver.save(sess, best_path)

def main(FLAGS):
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    word_vec_path = FLAGS.word_vec_path
    char_vec_path = FLAGS.char_vec_path
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    path_prefix = log_dir + "/SentenceMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    char_vocab = None
    if FLAGS.with_char and char_vec_path != "": char_vocab = Vocab(char_vec_path, fileformat='txt3')

    best_path = path_prefix + '.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    has_pre_trained_model = False
    
    if os.path.exists(best_path + ".index"):
        has_pre_trained_model = True
        print('Loading vocabs from a pre-trained model ...')
        label_vocab = Vocab(label_path, fileformat='txt2')
        if FLAGS.with_char: char_vocab = Vocab(char_path, fileformat='txt2')
    else:
        print('Collecting words, chars and labels ...')
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path)
        print('Number of words: {}'.format(len(all_words)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels, dim=2)
        label_vocab.dump_to_txt2(label_path)

        if FLAGS.with_char:
            print('Number of chars: {}'.format(len(all_chars)))
            if char_vec_path == "": char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=FLAGS.char_emb_dim)
            char_vocab.dump_to_txt2(char_path)
    
    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    if FLAGS.with_char: print('char_vocab shape is {}'.format(char_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    print("Number of labels: {}".format(num_classes))
    sys.stdout.flush()

    print('Build SentenceMatchDataStream ... ')
    trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab,
                                              isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    sys.stdout.flush()

    devDataStream = SentenceMatchDataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                    is_training=True, options=FLAGS, global_step=global_step)

        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                    is_training=False, options=FLAGS)

        initializer = tf.global_variables_initializer()
        initializer_local = tf.local_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            # if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer, feed_dict={train_graph.embedding: word_vocab.word_vecs})
        sess.run(initializer_local)
        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

        # training
        train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, FLAGS, best_path, label_vocab)

def enrich_options(options):
    if "in_format" not in options.__dict__:
        options.__dict__["in_format"] = 'tsv'
    if "filter_sizes" in options.__dict__:
        options.__dict__["filter_sizes"] = list(map(int, options.__dict__["filter_sizes"].split(",")))
    if "num_filters" in options.__dict__:    
        options.__dict__["num_filters"] = list(map(int, options.__dict__["num_filters"].split(",")))
    return options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--using_algo', type=str, default='bimpm', help='The using algorithm.')
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--out_path', type=str, help='Path to store the evaluation result.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--char_vec_path', type=str, help='Path the to pre-trained char vector model.')
    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l1', type=float, default=0.0, help='The coefficient of L1 regularizer.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--word_emb_dim', type=int, default=128, help='Number of dimension for word embeddings.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', help='Suffix of the model name.')
    parser.add_argument('--global_padding', default=False, help='Padding the sentence to the global max length, otherwith batch max length.', action='store_true')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--fix_char_vec', default=False, help='Fix pre-trained char embeddings during training.', action='store_true')
    parser.add_argument('--with_highway', default=False, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_full_match', default=False, help='With full matching.', action='store_true')
    parser.add_argument('--with_maxpool_match', default=False, help='With maxpooling matching', action='store_true')
    parser.add_argument('--with_attentive_match', default=False, help='With attentive matching', action='store_true')
    parser.add_argument('--with_max_attentive_match', default=False, help='With max attentive matching.', action='store_true')
    parser.add_argument('--with_char', default=False, help='With character-composed embeddings.', action='store_true')
    parser.add_argument('--with_f1_metric', default=False, help='F1 score Metric for the loss measure.')
    parser.add_argument('--filter_sizes', type=str, default='1,2,3', help='Filter sizes of the convolution layers.')
    parser.add_argument('--num_filters', type=str, default='10,10', help='Number of filters of the conv block A and conv block B.')
    parser.add_argument('--num_poolings', type=int, default=3, help='Number of pooling types of convoltion layers.')
    parser.add_argument('--lstm_out_type', type=str, default='end', help='The type of LSTM out, mean pooled of all steps or select the last step.')
    
    parser.add_argument('--config_path', type=str, help='Configuration file.')
    
    # print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    args, unparsed = parser.parse_known_args()
    if args.config_path is not None:
        print('Loading the configuration from ' + args.config_path)
        FLAGS = namespace_utils.load_namespace(args.config_path)
    else:
        FLAGS = args
    sys.stdout.flush()
    
    # enrich arguments to backwards compatibility
    FLAGS = enrich_options(FLAGS)

    main(FLAGS)

    # python SentenceMatchTrainer.py --config_path ../configs/quora.sample.config
