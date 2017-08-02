import tensorflow as tf

import sys
import os
import cPickle
from model_vaedyn import VAEDYN
import numpy as np

from train_vaedyn import next_batch

import dataset
import utils

np.random.seed(1234)
tf.set_random_seed(1234)

sample_mode = True

if len(sys.argv) == 1:
    #if no model folder is specified
    with open(os.path.join('save-vaedyn', 'config.pkl')) as f:
        saved_args = cPickle.load(f)
else:
    #use the specified model
    with open(os.path.join(sys.argv[1], 'config.pkl')) as f:
        saved_args = cPickle.load(f)

model = VAEDYN(saved_args, sample_mode)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.global_variables())

if len(sys.argv) == 1:
    #if no model folder is specified
    ckpt = tf.train.get_checkpoint_state('save-vaedyn')
else:
    ckpt = tf.train.get_checkpoint_state(sys.argv[1])

print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)

#prepare dataset
print 'Constructing dataset...'
image_seq_data = utils.extract_image_sequences(fname='bin/extracted_data_image_seq.pkl', only_digits=False)
print image_seq_data.shape
image_seq_dataset = dataset.construct_datasets(image_seq_data)

seq_samples = image_seq_dataset.test.next_batch(saved_args.batch_size)[0]

#encode a sequence of images
tol_len = 20
seeding_len = 14
prior_mus, prior_sigmas, enc_mus, enc_sigmas, final_state_c, final_state_h = model.encode(sess,saved_args, seq=seq_samples[0, :seeding_len, :])
#synthesis conditioned on the encoding
sample_data,mus = model.sample(sess,saved_args, num=tol_len-seeding_len, start=[seq_samples[0, seeding_len-1, :], (final_state_c, final_state_h)])

compl_data = np.vstack([seq_samples[0, :seeding_len, :], sample_data])

compl_diff = compl_data[seeding_len:tol_len, :] - seq_samples[0, seeding_len:tol_len, :]
#errors
square_errors = np.sum(np.sum((compl_diff)**2, axis=1), axis=0) / float((tol_len-seeding_len))
#crossentropy error
crossentropy_diff = seq_samples[0, seeding_len:tol_len, :] * np.log(1e-5 + compl_data[seeding_len:tol_len, :]) \
                    + (1-seq_samples[0, seeding_len:tol_len, :]) * np.log(1e-5 + 1 - compl_data[seeding_len:tol_len, :] )
crossentropy_errors = np.sum(np.sum(-crossentropy_diff, axis=1), axis=0) / float((tol_len-seeding_len))

print 'Square error: ', square_errors, 'CE error: ', crossentropy_errors
