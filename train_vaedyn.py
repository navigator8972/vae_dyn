#!/usr/bin/env python
import numpy as np
import tensorflow as tf

from copy import deepcopy
import argparse
import glob
import time
from datetime import datetime
import os
import cPickle

from model_vaedyn import VAEDYN

from matplotlib import pyplot as plt

import dataset
import utils

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict and sample functions
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
'''
np.random.seed(1234)
tf.set_random_seed(1234)

def next_batch(args):
    noise_switch=0.1
    t0 = np.random.randn(args.batch_size, 1, (2 * args.chunk_samples))
    mixed_noise = np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 * noise_switch #used to be 0.1
    #x = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #y = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    x = np.sin(2 * np.pi * (np.arange(args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 * noise_switch + mixed_noise*0.1
    y = np.sin(2 * np.pi * (np.arange(1, args.seq_length + 1)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 * noise_switch + mixed_noise*0.1

    y[:, :, args.chunk_samples:] = 0.
    x[:, :, args.chunk_samples:] = 0.
    return x, y


def train(args, model, ds=None):
    if ds is not None:
        n_samples = ds.train._data.shape[0]
        total_batch = int(n_samples / args.batch_size)
    else:
        total_batch = 1

    dirname = 'save-vaedyn'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)

        # <hyin/Sep-22nd-2017> do not use add_check_numerics_ops in while_loop: see tensorflow issue 2211
        # check = tf.add_check_numerics_ops()
        # merged = tf.merge_all_summaries()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Loaded model"
        start = time.time()

        lamb = 1
        sess.run(tf.assign(model.anneal_rate, 1.0*lamb))
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # sess.run(tf.assign(model.anneal_rate, np.amin([1.0*lamb, 0.001 + float(e) / args.num_epochs*lamb])))
            #<hyin/Jun-26th-2017> determine the probability to sample from prior
            #annealing: first always from posterior, then the possibility of sampling from prior increases
            # prior_prob = np.random.binomial(1, float(e) / args.num_epochs)
            # model.cell.prior_prob = prior_prob
            model.cell.prior_prob = float(e) / args.num_epochs
            # state = model.initial_state_c, model.initial_state_h
            # for b in xrange(100):
            #     x, y = next_batch(args)
            #     feed = {model.input_data: x, model.target_data: y}
            #     train_loss, _, cr, summary, sigmasummary, mu, input, target= sess.run(
            #             [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target],
            #                                                  feed)
            #     summary_writer.add_summary(summary, e * 100 + b)
            #     if (e * 100 + b) % args.save_every == 0 and ((e * 100 + b) > 0):
            #         checkpoint_path = os.path.join(dirname, 'model.ckpt')
            #         saver.save(sess, checkpoint_path, global_step=e * 100 + b)
            #         print "model saved to {}".format(checkpoint_path)
            #     end = time.time()
            #     print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
            #         .format(e * 100 + b,
            #                 args.num_epochs * 100,
            #                 e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0))
            #     start = time.time()

            for b in xrange(total_batch):
                if ds is None:
                    x, y = next_batch(args)
                else:
                    seq_samples = ds.train.next_batch(args.batch_size)[0]
                    x = seq_samples
                    y = seq_samples

                feed = {model.input_data: x, model.target_data: y}
                train_loss, _, summary= sess.run(
                        [model.cost, model.train_op, model.merged_summary],
                                                             feed)

                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}" \
                    .format(b + 1,
                            total_batch,
                            e+1, train_loss, end - start)
                start = time.time()

            # summary_writer.add_summary(summary, e)
            if (e + 1) % args.save_every == 0 and ((e + 1) > 0):
                checkpoint_path = os.path.join(dirname, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e * total_batch + b + 1)
                print "model saved to {}".format(checkpoint_path)
            # x, y = next_batch(args)
            #
            # train_loss, cr, summary, sigma, mu = model.train(sess, args, input_data=x,
            #         sequence_length=np.array([args.seq_length for i in range(args.batch_size)]), output_data=y, check=check, merged=merged)
            summary_writer.add_summary(summary, e)
            # if (e + 1) % args.save_every == 0 and ((e) > 0):
            #     checkpoint_path = os.path.join(dirname, 'model.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=e + b + 1)
            #     print "model saved to {}".format(checkpoint_path)
            # end = time.time()
            # print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
            #     .format(e,
            #             args.num_epochs,
            #             e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0))
            # start = time.time()

    return

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=3,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=3,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--chunk_samples', type=int, default=1,
                        help='number of samples per mdct chunk')
    parser.add_argument('--dim_size', type=int, default=1,
                        help='number of the input dimension')
    parser.add_argument('--use_cnn', type=str2bool, default=False,
                        help='use cnn as encoder/decoder')
    args = parser.parse_args()

    model = VAEDYN(args)

    #prepare dataset
    print 'Constructing dataset...'
    image_seq_data = utils.extract_image_sequences(fname='bin/extracted_data_image_seq.pkl', only_digits=False)
    # image_seq_data = np.array(utils.load_zipped_pickle('bin/rolling_dyn.zip'))
    print image_seq_data.shape
    image_seq_dataset = dataset.construct_datasets(image_seq_data)

    print 'Start TF training...'
    train(args, model, image_seq_dataset)
