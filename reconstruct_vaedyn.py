import tensorflow as tf

import os
import cPickle
from model_vaedyn import VAEDYN
import numpy as np

from train_vaedyn import next_batch

import dataset
import utils

sample_mode = True
np.random.seed(1234)
tf.set_random_seed(1234)

with open(os.path.join('save-vaedyn', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = VAEDYN(saved_args, sample_mode)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('save-vaedyn')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)

#prepare dataset
print 'Constructing dataset...'
image_seq_data = utils.extract_image_sequences(fname='bin/extracted_data_image_seq.pkl', only_digits=False)
print image_seq_data.shape
image_seq_dataset = dataset.construct_datasets(image_seq_data)

seq_samples = image_seq_dataset.train.next_batch(saved_args.batch_size)[0]

print seq_samples.shape
if sample_mode:
    reconstruct_data, prior_mus, prior_sigmas, enc_mus, enc_sigmas, reconstr_err, final_state_c, final_state_h = model.reconstruct(sess,saved_args, seq=seq_samples[0])
else:
    reconstruct_data, prior_mus, prior_sigmas, enc_mus, enc_sigmas, reconstr_err, final_state_c, final_state_h = model.reconstruct(sess,saved_args, seq=seq_samples)

print 'diff prior/enc mu:', np.linalg.norm(prior_mus-enc_mus, axis=1)/np.linalg.norm(prior_mus, axis=1)
print 'diff prior/enc sigma:', np.linalg.norm(prior_sigmas-enc_sigmas, axis=1)/np.linalg.norm(prior_sigmas, axis=1)
print 'reconstruct error:', reconstr_err
