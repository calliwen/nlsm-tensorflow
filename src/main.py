#/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import sys
import jieba as jb
import tensorflow as tf

import namespace_utils
from vocab_utils import Vocab
from SentenceMatchModelGraph import SentenceMatchModelGraph
from SentenceMatchDataStream import SentenceMatchDataStream


def data_parser(inpath, outpath):
    jb.load_userdict('user_dict.txt')
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            sen1 = ' '.join(jb.cut(sen1.replace('***','')))
            sen2 = ' '.join(jb.cut(sen2.replace('***','')))
            fout.write('0' + '\t' + sen1.encode('utf-8') + '\t' + sen2.encode('utf-8') + '\t' + lineno + '\n')

def evaluation(sess, valid_graph, devDataStream, outpath=None, label_vocab=None):
    result_json = {}
    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=False)
        [predictions] = sess.run([valid_graph.predictions], feed_dict=feed_dict)
        for i in range(cur_batch.batch_size):
            (_, _, _, _, _, _, _, _, cur_ID) = cur_batch.instances[i]
            result_json[cur_ID] = {
                "ID": cur_ID,
                "prediction": label_vocab.getWord(predictions[i]),
            }
    return result_json

if __name__ == '__main__':
    data_parser(sys.argv[1], './tmp.tsv')

    in_path = './tmp.tsv'
    out_path = sys.argv[2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, default='./data_quora/logs_bimpm/SentenceMatch.quora', help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, default=in_path, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, default=out_path, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default='./wordvec.txt', help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()
    
    # load the configuration file
    print('Loading configurations.')
    options = namespace_utils.load_namespace(args.model_prefix + ".config.json")

    if args.word_vec_path is None: args.word_vec_path = options.word_vec_path

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(args.word_vec_path, fileformat='txt3')
    label_vocab = Vocab(args.model_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    char_vocab = None
    if options.with_char:
        char_vocab = Vocab(args.model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    
    print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchDataStream(args.in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                            label_vocab=label_vocab,
                                            isShuffle=False, isLoop=True, isSort=True, options=options)
    print('Number of instances in devDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    best_path = args.model_prefix + ".best.model"
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  is_training=False, options=options)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        print("Restoring model from " + best_path)
        saver.restore(sess, best_path)
        print("DONE!")
        
        res = evaluation(sess, valid_graph, testDataStream, outpath=None,
                                              label_vocab=label_vocab)
        with open(sys.argv[1], 'r') as fin, open(sys.argv[2], 'w') as fout:
            for line in fin:
                lineno, sen1, sen2 = line.strip().split('\t')
                fout.write(res[lineno]['ID'] + '\t' + res[lineno]['prediction'] + '\n')

#  bash run.sh test.tsv out.tsv