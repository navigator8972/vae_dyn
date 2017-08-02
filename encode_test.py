'''
A script to test the completion of image sequences
'''
import tensorflow as tf

import sys
import os
import cPickle
from model_vaedyn import VAEDYN
import numpy as np

from train_vaedyn import next_batch

import dataset
import utils



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
saver = tf.train.Saver(tf.global_variables())
# saver = tf.train.Saver()
sess = tf.InteractiveSession()


if len(sys.argv) == 1:
    #if no model folder is specified
    ckpt = tf.train.get_checkpoint_state('save-vaedyn')

    model_checkpoint_path = ckpt.model_checkpoint_path
elif len(sys.argv) == 2:
    ckpt = tf.train.get_checkpoint_state(sys.argv[1])
    model_checkpoint_path = ckpt.model_checkpoint_path
elif len(sys.argv) == 3:
    model_checkpoint_path = os.path.join(sys.argv[1], sys.argv[2])

print "loading model: ",model_checkpoint_path

saver.restore(sess, model_checkpoint_path)


#prepare seed to have stable shuffle
np.random.seed(1234)
tf.set_random_seed(1234)

#prepare dataset
print 'Constructing dataset...'
image_seq_data = utils.extract_image_sequences(fname='bin/extracted_data_image_seq.pkl', only_digits=False)
print image_seq_data.shape
image_seq_dataset = dataset.construct_datasets(image_seq_data)
#encode a sequence of images
tol_len = 20
seeding_len = 14

seq_samples = image_seq_dataset.test._data

compl_samples = seq_samples.copy()

for idx, seq_sample in enumerate(seq_samples):
    print 'Processing test sample {0}...'.format(idx)
    seq_sample_formatted = np.array([seq_sample])
    prior_mus, prior_sigmas, enc_mus, enc_sigmas, final_state_c, final_state_h = model.encode(sess,saved_args, seq=seq_sample_formatted[0, :seeding_len, :])
    #synthesis conditioned on the encoding
    sample_data,mus = model.sample(sess,saved_args, num=tol_len-seeding_len, start=[seq_sample_formatted[0, seeding_len-1, :], (final_state_c, final_state_h)])

    # compl_data = np.concatenate([seq_sample_formatted[:, :seeding_len, :], sample_data], axis=1)
    compl_samples[idx, seeding_len:tol_len, :] = sample_data

compl_diff = compl_samples[:, seeding_len:tol_len, :] - seq_samples[:, seeding_len:tol_len, :]
#errors
squre_errors = np.sum(np.sum((compl_diff)**2, axis=2), axis=1) / float((tol_len-seeding_len))

crossentropy_diff = seq_samples[:, seeding_len:tol_len, :] * np.log(1e-5 + compl_samples[:, seeding_len:tol_len, :]) \
                    + (1-seq_samples[:, seeding_len:tol_len, :]) * np.log(1e-5 + 1 - compl_samples[:, seeding_len:tol_len, :] )
crossentropy_errors = np.sum(np.sum(-crossentropy_diff, axis=2), axis=1) / float((tol_len-seeding_len))

mean_square_errors = np.mean(squre_errors)
std_square_errors = np.std(squre_errors)

mean_ce_errors = np.mean(crossentropy_errors)
std_ce_errors = np.std(crossentropy_errors)

print 'Square Error Mean: ', mean_square_errors, '; Std: ', std_square_errors
print 'Cross Entropy Error Mean: ', mean_ce_errors, '; Std: ', std_ce_errors
