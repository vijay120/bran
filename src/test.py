import time
import random
import re
import sys
import numpy as np
import gzip
from collections import defaultdict
from src.data_utils import *
from src.tf_utils import *
from src.models.classifier_models import *
from src.evaluation.ner_eval import ner_eval
from src.evaluation.relation_eval import relation_eval
from src.evaluation.export_predictions import *
from src.feed_dicts import *
FLAGS = tf.app.flags.FLAGS


def main(argv):
    if ('transformer' in FLAGS.text_encoder or 'glu' in FLAGS.text_encoder) and FLAGS.token_dim == 0:
        FLAGS.token_dim = FLAGS.embed_dim - (2 * FLAGS.position_dim)
        # print flags:values in alphabetical order
    print ('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))

    if FLAGS.vocab_dir == '':
        print('Error: Must supply input data generated from tsv_to_tfrecords.py')
        sys.exit(1)

    position_vocab_size = (2 * FLAGS.max_seq)

    # read in str <-> int vocab maps
    with open(FLAGS.vocab_dir + '/rel.txt', 'r') as f:
        kb_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        kb_id_str_map = {i: s for s, i in kb_str_id_map.iteritems()}
        kb_vocab_size = FLAGS.kb_vocab_size
    with open(FLAGS.vocab_dir + '/token.txt', 'r') as f:
        token_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        if FLAGS.start_end:
            if '<START>' not in token_str_id_map: token_str_id_map['<START>'] = len(token_str_id_map)
            if '<END>' not in token_str_id_map: token_str_id_map['<END>'] = len(token_str_id_map)
        token_id_str_map = {i: s for s, i in token_str_id_map.iteritems()}
        token_vocab_size = len(token_id_str_map)

    with open(FLAGS.vocab_dir + '/entities.txt', 'r') as f:
        entity_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        entity_id_str_map = {i: s for s, i in entity_str_id_map.iteritems()}
        entity_vocab_size = len(entity_id_str_map)
    with open(FLAGS.vocab_dir + '/ep.txt', 'r') as f:
        ep_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        ep_id_str_map = {i: s for s, i in ep_str_id_map.iteritems()}
        ep_vocab_size = len(ep_id_str_map)

    ep_kg_labels = None
    if FLAGS.kg_label_file != '':
        kg_in_file = gzip.open(FLAGS.kg_label_file, 'rb') if FLAGS.kg_label_file.endswith('gz') else open(
            FLAGS.kg_label_file, 'r')
        lines = [l.strip().split() for l in kg_in_file.readlines()]
        eps = [('%s::%s' % (l[0], l[1]), l[2]) for l in lines]
        ep_kg_labels = defaultdict(set)
        [ep_kg_labels[ep_str_id_map[_ep]].add(pid) for _ep, pid in eps if _ep in ep_str_id_map]
        print('Ep-Kg label map size %d ' % len(ep_kg_labels))
        kg_in_file.close()

    label_weights = None
    if FLAGS.label_weights != '':
        with open(FLAGS.label_weights, 'r') as f:
            lines = [l.strip().split('\t') for l in f]
            label_weights = {kb_str_id_map[k]: float(v) for k, v in lines}

    model_type = MultiLabelClassifier
    ner_label_id_str_map = {}
    ner_label_str_id_map = {}
    ner_label_vocab_size = 1
    e1_e2_ep_map = {}  # {(entity_str_id_map[ep_str.split('::')[0]], entity_str_id_map[ep_str.split('::')[1]]): ep_id
    # for ep_id, ep_str in ep_id_str_map.iteritems()}
    ep_e1_e2_map = {}  # {ep: e1_e2 for e1_e2, ep in e1_e2_ep_map.iteritems()}

    word_embedding_matrix = load_pretrained_embeddings(token_str_id_map, FLAGS.embeddings, FLAGS.token_dim,
                                                       token_vocab_size)
    entity_embedding_matrix = load_pretrained_embeddings(entity_str_id_map, FLAGS.entity_embeddings, FLAGS.embed_dim,
                                                         entity_vocab_size)

    string_int_maps = {'kb_str_id_map': kb_str_id_map, 'kb_id_str_map': kb_id_str_map,
                       'token_str_id_map': token_str_id_map, 'token_id_str_map': token_id_str_map,
                       'entity_str_id_map': entity_str_id_map, 'entity_id_str_map': entity_id_str_map,
                       'ep_str_id_map': ep_str_id_map, 'ep_id_str_map': ep_id_str_map,
                       'ner_label_str_id_map': ner_label_str_id_map, 'ner_label_id_str_map': ner_label_id_str_map,
                       'e1_e2_ep_map': e1_e2_ep_map, 'ep_e1_e2_map': ep_e1_e2_map, 'ep_kg_labels': ep_kg_labels,
                       'label_weights': label_weights}


    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        random.seed(FLAGS.random_seed)

        if FLAGS.doc_filter:
            train_percent = FLAGS.train_dev_percent
            with open(FLAGS.doc_filter, 'r') as f:
                doc_filter_ids = [l.strip() for l in f]
            shuffle(doc_filter_ids)
            split_idx = int(len(doc_filter_ids) * train_percent)
            dev_ids, train_ids = set(doc_filter_ids[:split_idx]), set(doc_filter_ids[split_idx:])
            # ids in dev_ids will be filtered from dev, same for train_ids
            print('Splitting dev data %d documents for train and %d documents for dev' % (len(dev_ids), len(train_ids)))
        else:
            dev_ids, train_ids = None, None

        positive_test_batcher = InMemoryBatcher(FLAGS.positive_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.positive_test else None
        negative_test_batcher = InMemoryBatcher(FLAGS.negative_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.negative_test else None
        positive_test_test_batcher = InMemoryBatcher(FLAGS.positive_test_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.positive_test_test else None


        # have seperate batchers for positive and negative train/test
        batcher = InMemoryBatcher if FLAGS.in_memory else Batcher
        positive_test_batcher = InMemoryBatcher(FLAGS.positive_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.positive_test else None
        negative_test_batcher = InMemoryBatcher(FLAGS.negative_test, 1, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.negative_test else None

        model = model_type(ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                           ner_label_vocab_size, word_embedding_matrix, entity_embedding_matrix, string_int_maps, FLAGS)

        # restore only variables that exist in the checkpoint - needed to pre-train big models with small models
        if FLAGS.load_model != '':
            reader = tf.train.NewCheckpointReader(FLAGS.load_model)
            cp_list = set([key for key in reader.get_variable_to_shape_map()])
            # if variable does not exist in checkpoint or sizes do not match, dont load
            r_vars = [k for k in tf.global_variables() if k.name.split(':')[0] in cp_list
                      and k.get_shape() == reader.get_variable_to_shape_map()[k.name.split(':')[0]]]
            if len(cp_list) != len(r_vars):
                print('[Warning]: not all variables loaded from file')
                # print('\n'.join(sorted(set(cp_list)-set(r_vars))))
            saver = tf.train.Saver(var_list=r_vars)
        else:
            saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=FLAGS.logdir if FLAGS.save_model != '' else None,
                                 global_step=model.global_step,
                                 saver=None,
                                 save_summaries_secs=0,
                                 save_model_secs=0, )

        with sv.managed_session(FLAGS.master,
                                config=tf.ConfigProto(
                                    # log_device_placement=True,
                                    allow_soft_placement=True
                                )) as sess:

            if positive_test_batcher: positive_test_batcher.load_all_data(sess, doc_filter=dev_ids)
            if negative_test_batcher: negative_test_batcher.load_all_data(sess, doc_filter=dev_ids)

            if FLAGS.load_model != '':
                print("Deserializing model: %s" % FLAGS.load_model)
                saver.restore(sess, FLAGS.load_model)

            import pdb; pdb.set_trace()
            relation_eval(sess, model, FLAGS, positive_test_batcher, negative_test_batcher, string_int_maps)



if __name__ == '__main__':
    tf.app.flags.DEFINE_string('vocab_dir', '', 'tsv file containing string data')
    tf.app.flags.DEFINE_string('kb_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_dist_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_dist_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_test', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_test', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('positive_test_test', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('negative_test_test', '',
                                'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('ner_train', '',
                               'file pattern of proto buffers generated from ../src/processing/ner_to_tfrecords.py')
    tf.app.flags.DEFINE_string('ner_test', '',
                               'file pattern of proto buffers generated from ../src/processing/ner_to_tfrecords.py')
    tf.app.flags.DEFINE_string('embeddings', '', 'pretrained word embeddings')
    tf.app.flags.DEFINE_string('entity_embeddings', '', 'pretrained entity embeddings')
    tf.app.flags.DEFINE_string('fb15k_dir', '', 'directory containing fb15k tsv files')
    tf.app.flags.DEFINE_string('nci_dir', '', 'directory containing nci tsv files')
    tf.app.flags.DEFINE_string('noise_dir', '',
                               'directory containing fb15k noise files generated from src/util/generate_noise.py')
    tf.app.flags.DEFINE_string('candidate_file', '', 'candidate file for tac evaluation')
    tf.app.flags.DEFINE_string('variance_file', '', 'variance file in candidate file format')
    tf.app.flags.DEFINE_string('type_file', '', 'tsv mapping entities to types')
    tf.app.flags.DEFINE_string('kg_label_file', '', '13 col tsv for mapping eps -> kg relations')
    tf.app.flags.DEFINE_string('label_weights', '', 'weight examples for unbalanced labels')
    tf.app.flags.DEFINE_string('logdir', '', 'save logs and models to this dir')
    tf.app.flags.DEFINE_string('load_model', '', 'path to saved model to load')
    tf.app.flags.DEFINE_string('save_model', '', 'name of file to serialize model to')
    tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
    tf.app.flags.DEFINE_string('loss_type', 'softmax', 'optimizer to use')
    tf.app.flags.DEFINE_string('model_type', 'd', 'optimizer to use')
    tf.app.flags.DEFINE_string('text_encoder', 'lstm', 'optimizer to use')
    # todo: make compatitble with cnn and transformer as width:dilation:take
    tf.app.flags.DEFINE_string('layer_str', '1:1,5:1,1:1', 'transformer feed-forward layers (width:dilation)')
    # tf.app.flags.DEFINE_string('layer_str', '1:false,2:false,1:true', 'cnn layers (dilation:take)')
    tf.app.flags.DEFINE_string('variance_type', 'divide', 'type of variance model to use')
    tf.app.flags.DEFINE_string('mode', 'train', 'train, evaluate, analyze')
    tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
    tf.app.flags.DEFINE_string('doc_filter', '', 'file to dev doc ids to split between train and test')
    tf.app.flags.DEFINE_string('thresholds', '.5,.6,.7,.75,.8,.85,.9,.95,.975', 'thresholds for prediction')
    tf.app.flags.DEFINE_string('null_label', '0', 'index of negative label')
    tf.app.flags.DEFINE_string('export_file', '', 'export predictions to this file in biocreative VI format')

    tf.app.flags.DEFINE_boolean('norm_entities', False, 'normalize entitiy vectors to have unit norm')
    tf.app.flags.DEFINE_boolean('bidirectional', False, 'bidirectional lstm')
    tf.app.flags.DEFINE_boolean('use_tanh', False, 'use tanh')
    tf.app.flags.DEFINE_boolean('use_peephole', False, 'use peephole connections in lstm')
    tf.app.flags.DEFINE_boolean('max_pool', False, 'max pool hidden states of lstm, else take last')
    tf.app.flags.DEFINE_boolean('in_memory', False, 'load data in memory')
    tf.app.flags.DEFINE_boolean('reset_variance', False, 'reset loaded variance projection matrices')
    tf.app.flags.DEFINE_boolean('percentile', False, 'variance weight based off of percentile')
    tf.app.flags.DEFINE_boolean('semi_hard', False, 'use semi hard negative sample selection')
    tf.app.flags.DEFINE_boolean('verbose', False, 'additional logging')
    tf.app.flags.DEFINE_boolean('freeze', False, 'freeze row and column params')
    tf.app.flags.DEFINE_boolean('freeze_noise', False, 'freeze row and column params')
    tf.app.flags.DEFINE_boolean('mlp', False, 'mlp instead of linear for classification')
    tf.app.flags.DEFINE_boolean('debug', False, 'flags for testing')
    tf.app.flags.DEFINE_boolean('start_end', False, 'add start and end tokens to examples')
    tf.app.flags.DEFINE_boolean('filter_pad', False, 'zero out pad token embeddings and attention')
    tf.app.flags.DEFINE_boolean('anneal_ner', False, 'anneal ner prob as training goes on')
    tf.app.flags.DEFINE_boolean('tune_macro_f', False, 'early stopping based on macro F, else micro F')

    # tac eval
    tf.app.flags.DEFINE_boolean('center_only', False, 'only take center in tac eval')
    tf.app.flags.DEFINE_boolean('arg_entities', False, 'replaced entities with arg wildcards')
    tf.app.flags.DEFINE_boolean('norm_digits', True, 'norm digits in tac eval')

    tf.app.flags.DEFINE_float('lr', .01, 'learning rate')
    tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon for adam optimizer')
    tf.app.flags.DEFINE_float('beta2', 0.999, 'beta2 adam optimizer')
    tf.app.flags.DEFINE_float('lr_decay_steps', 25000, 'anneal learning rate every k steps')
    tf.app.flags.DEFINE_float('lr_decay_rate', .75, 'anneal learning rate every k steps')
    tf.app.flags.DEFINE_float('margin', 1.0, 'margin for hinge loss')
    tf.app.flags.DEFINE_float('l2_weight', 0.0, 'weight for l2 loss')
    tf.app.flags.DEFINE_float('dropout_loss_weight', 0.0, 'weight for dropout loss')
    tf.app.flags.DEFINE_float('clip_norm', 1, 'clip gradients to have norm <= this')
    tf.app.flags.DEFINE_float('text_weight', 1.0, 'weight for text updates')
    tf.app.flags.DEFINE_float('ner_weight', 1.0, 'weight for text updates')
    tf.app.flags.DEFINE_float('ner_prob', 0.0, 'probability of drawing a text batch vs kb batch')
    tf.app.flags.DEFINE_float('text_prob', .5, 'probability of drawing a text batch vs kb batch')
    tf.app.flags.DEFINE_float('pos_prob', .5, 'probability of drawing a positive example')
    tf.app.flags.DEFINE_float('f_beta', 1, 'evaluate using the f_beta metric')
    tf.app.flags.DEFINE_float('noise_std', 0, 'add noise to gradients from 0 mean gaussian with this std')
    tf.app.flags.DEFINE_float('train_dev_percent', .6, 'use this portion of dev as additional train data')

    tf.app.flags.DEFINE_float('variance_min', 1.0, 'weight of variance penalty')
    tf.app.flags.DEFINE_float('variance_max', 99.9, 'weight of variance penalty')
    tf.app.flags.DEFINE_float('variance_delta', 0.0, 'increase variance weight by this value each step')
    tf.app.flags.DEFINE_float('pos_noise', 0.0, 'increase variance weight by this value each step')
    tf.app.flags.DEFINE_float('neg_noise', 0.0, 'increase variance weight by this value each step')

    tf.app.flags.DEFINE_float('word_dropout', .9, 'dropout keep probability for word embeddings')
    tf.app.flags.DEFINE_float('word_unk_dropout', 1.0, 'dropout keep probability for word embeddings')
    tf.app.flags.DEFINE_float('pos_unk_dropout', 1.0, 'dropout keep probability for position embeddings')
    tf.app.flags.DEFINE_float('lstm_dropout', 1.0, 'dropout keep probability for lstm output before projection')
    tf.app.flags.DEFINE_float('final_dropout', 1.0, 'dropout keep probability for final row and column representations')

    tf.app.flags.DEFINE_integer('pattern_dropout', 10, 'take this many mentions for rowless')
    tf.app.flags.DEFINE_integer('pos_count', 2206761, 'number of positive training examples')
    tf.app.flags.DEFINE_integer('neg_count', 20252779, 'number of negative training examples')
    tf.app.flags.DEFINE_integer('kb_vocab_size', 237, 'learning rate')
    tf.app.flags.DEFINE_integer('text_batch', 32, 'batch size')
    tf.app.flags.DEFINE_integer('eval_batch', 32, 'batch size')
    tf.app.flags.DEFINE_integer('kb_batch', 4096, 'batch size')
    tf.app.flags.DEFINE_integer('ner_batch', 128, 'batch size')
    tf.app.flags.DEFINE_integer('token_dim', 250, 'token dimension')
    tf.app.flags.DEFINE_integer('lstm_dim', 2048, 'lstm internal dimension')
    tf.app.flags.DEFINE_integer('embed_dim', 100, 'row/col embedding dimension')
    tf.app.flags.DEFINE_integer('position_dim', 5, 'position relative to entities in lstm embedding')
    tf.app.flags.DEFINE_integer('text_epochs', 100, 'train for this many text epochs')
    tf.app.flags.DEFINE_integer('kb_epochs', 100, 'train for this many kb epochs')
    tf.app.flags.DEFINE_integer('kb_pretrain', 0, 'pretrain kb examples for this many steps')
    tf.app.flags.DEFINE_integer('block_repeats', 1, 'apply iterated blocks this many times')
    tf.app.flags.DEFINE_integer('alternate_var_train', 0,
                                'alternate between variance and rest optimizers every k steps')
    tf.app.flags.DEFINE_integer('log_every', 10, 'log every k steps')
    tf.app.flags.DEFINE_integer('eval_every', 10000, 'eval every k steps')
    tf.app.flags.DEFINE_integer('max_steps', -1, 'stop training after this many total steps')
    tf.app.flags.DEFINE_integer('max_seq', 1, 'maximum sequence length')
    tf.app.flags.DEFINE_integer('max_decrease_epochs', 33, 'stop training early if eval doesnt go up')
    tf.app.flags.DEFINE_integer('num_classes', 4, 'number of classes for multiclass classifier')
    tf.app.flags.DEFINE_integer('neg_samples', 200, 'number of negative samples')
    tf.app.flags.DEFINE_integer('random_seed', 1111, 'random seed')
    tf.app.flags.DEFINE_integer('analyze_errors', 0, 'print out error analysis for K examples per type and exit')


tf.app.run()