import tensorflow as tf

import sys
import os
import cPickle
from model_vaedyn import VAEDYN
import numpy as np
import time

from train_vrnn import next_batch
import utils

if len(sys.argv) == 1:
    with open(os.path.join('save-vaedyn', 'config.pkl')) as f:
        saved_args = cPickle.load(f)
else:
    with open(os.path.join(os.path.dirname(sys.argv[1]), 'config.pkl')) as f:
        saved_args = cPickle.load(f)

model = VAEDYN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.global_variables())

#use checkpoint if not specified
if len(sys.argv) == 1:
    ckpt = tf.train.get_checkpoint_state('save-vaedyn')
    print "loading model: ",ckpt.model_checkpoint_path

    saver.restore(sess, ckpt.model_checkpoint_path)

    sample_data,zs = model.sample(sess,saved_args, num=20)
else:
    print "loading model: ",sys.argv[1]
    saver.restore(sess, sys.argv[1])

    sample_data,zs = model.sample(sess,saved_args, num=16)
