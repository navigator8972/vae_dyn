import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    sequential_mode = len(shape) > 2    #need to deal with sequential case that input_ is batch x seq_len x dim
                                        #tensorflow seems not supporting tensor product right now, use reshape
    if sequential_mode:
        reshaped_input = tf.reshape(input_, (-1, shape[-1]))
    else:
        reshaped_input = input_

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable("matrix", [shape[-1], output_size], tf.float32, tf.random_uniform_initializer(-stddev, stddev))

        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        
        ret = tf.matmul(reshaped_input, matrix) + bias

        if sequential_mode:
            reshaped_ret = tf.reshape(ret, (shape[0], shape[1], -1))
        else:
            reshaped_ret = ret

        if with_w:
            return reshaped_ret, matrix, bias
        else:
            return reshaped_ret + bias

def nonlinear(input_, n_units=10, n_layers=1, nonlinearity_type='relu', scope=None, stddev=0.02, bias_start=0.0):
    """
    Wrapper to represent a nonlinear transformation phi(x)
    """
    if type(n_units) is not list:
        n_latent_units = [n_units for i in range(n_layers)]
    else:
        n_latent_units = n_units
        n_layers = len(n_units)

    shape = input_.get_shape().as_list()
    sequential_mode = len(shape) > 2    #need to deal with sequential case that input_ is batch x seq_len x dim
                                        #tensorflow seems not supporting tensor product right now, use reshape
    if sequential_mode:
        reshaped_input = tf.reshape(input_, (-1, shape[-1]))
    else:
        reshaped_input = input_    

    with tf.variable_scope(scope or 'nonlinear'):
        last_output_dim = shape[-1]
        last_output = reshaped_input
        for i in range(n_layers):
            matrix = tf.get_variable('matrix_{0}'.format(i), [last_output_dim, n_latent_units[i]], tf.float32,
                                    tf.random_uniform_initializer(-stddev, stddev))
            bias = tf.get_variable("bias_{0}".format(i), [n_latent_units[i]], initializer=tf.constant_initializer(bias_start))
            last_output = tf.matmul(last_output, matrix) + bias

            if nonlinearity_type == 'relu':
                last_output = tf.nn.relu(last_output)

            last_output_dim = n_latent_units[i]

        if sequential_mode:
            reshaped_last_output = tf.reshape(last_output, (shape[0], shape[1], -1))
        else:
            reshaped_last_output = last_output
    return reshaped_last_output

from tensorflow.contrib import slim

def conv_encoder(input_, n_units=10, feature_shape=(28, 28, 1), is_training=True):
    #convoluational layers to process input pixels
    #input_: batch_size * dim_size
    with tf.variable_scope('convolution'):
        net = tf.reshape(input_, [-1, feature_shape[0], feature_shape[1], feature_shape[2]])    #batch_size*seq_len, horizon, vertical, channels
        with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.elu
                        ):
            net = slim.conv2d(net, 32, 5, stride=2)                                                 #14x14x32
            net = slim.conv2d(net, 64, 5, stride=2)                                                 #7x7x64
            net = slim.conv2d(net, 128, 5, padding='VALID')                                         #3x3x128
            net = slim.conv2d(net, 128, 3, padding='VALID')                                         #1x1x128
            # net = slim.dropout(net, 0.9)
        net = tf.reshape(net, [-1, 128])
        enc_hidden = linear(net, n_units)                                                           #for mu and sigma
    return enc_hidden

def deconv_decoder(input_, feature_shape=(28, 28, 1), is_training=True):
    with tf.variable_scope('deconvolution'):
        net = tf.reshape(input_, [-1, 1, 1, input_.get_shape().as_list()[-1]])                                  #batch_size*seq_len, 1, 1, channels (would be a concatenated one)
        with slim.arg_scope([slim.conv2d_transpose],
                activation_fn=tf.nn.elu
                ):
            net = slim.conv2d_transpose(net, 128, 3, padding='VALID')                               #3x3x128
            net = slim.conv2d_transpose(net, 64, 5, padding='VALID')                                #5x5x64
            net = slim.conv2d_transpose(net, 32, 5, stride=2)                                       #13x13x32
            net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)                    #28x28x1
        dec_hidden = tf.reshape(net, [input_.get_shape().as_list()[0], -1, feature_shape[0]*feature_shape[1]*feature_shape[2]])
    return dec_hidden

class VartiationalRNNEncoder(tf.contrib.rnn.RNNCell):
    """Variational RNN encoder."""

    def __init__(self, x_dim, h_dim, z_dim = 100, renc=True, use_cnn=False, is_training=True):
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = z_dim
        self.n_prior_hidden = z_dim
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True, initializer=tf.orthogonal_initializer(gain=1.0))
        self.prior_prod = 0.0
        self.renc = renc
        self.use_cnn = use_cnn
        self.is_training = is_training

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return (self.n_z, self.n_z, self.n_z, self.n_z, self.n_z, self.n_z)

    def zero_state(self, batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)

    def __call__(self, x, state, scope=None):
        if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
            c = state.c
            h = state.h
        else:
            c, h = state
        
        if not self.use_cnn:
            with tf.variable_scope("phi_x"):
                x_1 = nonlinear(x, n_units=[400, 400, 200], n_layers=3)

            with tf.variable_scope("hidden"):
                if self.renc:
                    enc_hidden = nonlinear(tf.concat((x_1, h), 1), n_units=200, n_layers=1)
                else:
                    enc_hidden = nonlinear(x_1, n_units=200, n_layers=1)
                with tf.variable_scope("mu"):
                    enc_mu    = linear(enc_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.exp(linear(enc_hidden, self.n_z, bias_start=0.0))
        else:
            enc_cnn = conv_encoder(input_=x, n_units=128, feature_shape=(28, 28, 1), is_training=self.is_training)
            with tf.variable_scope("hidden"):
                concat_rep = tf.concat((enc_cnn, h), 1)
                enc_hidden = nonlinear(concat_rep, n_units=128, n_layers=1)
                with tf.variable_scope("mu"):
                    enc_mu    = linear(enc_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.exp(linear(enc_hidden, self.n_z, bias_start=0.0))

        self.eps = tf.random_normal((x.get_shape().as_list()[0], self.n_z), 0.0, 1.0, dtype=tf.float32)

        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma
        self.enc_mu_post = enc_mu
        self.enc_sigma_post = enc_sigma
        
        with tf.variable_scope("z"):
            self.z = tf.add(self.enc_mu_post, tf.multiply(self.enc_sigma_post, self.eps))

        with tf.variable_scope("phi_z"):
            z_1 = nonlinear(self.z, self.n_z_1, n_layers=1)

        output, new_state = self.lstm(z_1, state, scope="LSTMCell")

        return (self.enc_mu, self.enc_sigma, self.enc_mu_post, self.enc_sigma_post, self.z, h), new_state

class VAEDYN():
    def __init__(self, args, sample=False):

        def tf_cross_entropy(y, mu):
            #note here y and mu are reshaped with the batch and seq_length dimensions merged
            #so reduce_sum(x, 1) means summing up the difference of pixels
            reconstr_loss = \
                -tf.reduce_sum(y * tf.log(1e-5 + mu)
                               + (1-y) * tf.log(1e-5 + 1 - mu),
                               1)
            return reconstr_loss

        def tf_square(y, mu):
            result = tf.reduce_sum(tf.square(tf.subtract(y, mu)), 1)
            return result

        def tf_normal(y, mu, s, rho):
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10,tf.square(s))
                norm = tf.subtract(y[:,:], mu)
                z = tf.div(tf.square(norm), ss)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                result = tf.reduce_sum(z+denom_log, 1)/2
            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-8,sigma_2),name='log_sigma_2')
                  - 2 * tf.log(tf.maximum(1e-8,sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-8,(tf.square(sigma_2))) - 1
                ), 1)

        def tf_kl_gaussunisotropic(mu1, sigma_1):
            with tf.variable_scope("kl_gaussunistropic"):
                return -0.5 * tf.reduce_sum(1 + 2*tf.log(tf.maximum(1e-8,sigma_1),name='log_sigma_1')
                                                   - tf.square(mu1)
                                                   - tf.square(sigma_1), 1)

        def tf_kl_smooth(mu, sigma, args):
            with tf.variable_scope('kl_smooth'):
                mu_split = tf.split(mu, args.batch_size, 0)
                sigma_split = tf.split(sigma, args.batch_size, 0)

                mu_1 = tf.slice(mu_split, [0, 0, 0], [-1, args.seq_length-1, -1])
                sigma_1 = tf.slice(sigma_split, [0, 0, 0], [-1, args.seq_length-1, -1])
                mu_2 = tf.slice(mu_split, [0, 1, 0], [-1, -1, -1])
                sigma_2 = tf.slice(sigma_split, [0, 1, 0], [-1, -1, -1])

                mu_1_reshape = tf.reshape(mu_1, [(args.seq_length-1)*args.batch_size, -1])
                sigma_1_reshape = tf.reshape(sigma_1, [(args.seq_length-1)*args.batch_size, -1])
                mu_2_reshape = tf.reshape(mu_2, [(args.seq_length-1)*args.batch_size, -1])
                sigma_2_reshape = tf.reshape(sigma_2, [(args.seq_length-1)*args.batch_size, -1])
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-8,sigma_2_reshape),name='log_sigma_2')
                  - 2 * tf.log(tf.maximum(1e-8,sigma_1_reshape),name='log_sigma_1')
                  + (tf.square(sigma_1_reshape) + tf.square(mu_1_reshape - mu_2_reshape)) / tf.maximum(1e-8,(tf.square(sigma_2_reshape))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, prior_mu, prior_sigma, y, anneal_rate, args):
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            kl_unitropic_loss = tf_kl_gaussunisotropic(enc_mu, enc_sigma)

            square_loss = tf_square(y, dec_mu)
            cross_ent_loss = tf_cross_entropy(y, dec_mu)

            # smooth_reg_loss = tf_kl_smooth(prior_mu, prior_sigma, args) * 0

            #<hyin/Apr-04-2017> also try an isotropic regularization on the prior z, would that imply a smooth prior?
            return tf.reduce_mean(anneal_rate * kl_loss + cross_ent_loss)  #+ tf.reduce_mean(kl_unitropic_loss) #original vrnn needs to disable this

        self.args = args
        if sample:
            args.batch_size = 1
            args.seq_length = 1

        self.is_sample = sample
        self.is_training = not self.is_sample

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.dim_size], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.dim_size],name = 'target_data')
        
        # rnn cell for encoder
        cell = VartiationalRNNEncoder(args.dim_size, args.rnn_size, args.latent_size, True, args.use_cnn, self.is_training)
        self.cell = cell
        #zero state
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        self.initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_state_c,self.initial_state_h)

        self.input_sequence_len = tf.placeholder(dtype=tf.int32, shape=[args.batch_size])

        input_seq_len = [args.seq_length] * args.batch_size
        
        if self.is_sample:
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, args.dim_size])

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(inputs, args.seq_length, 0) # n_steps * (batch_size, n_input)

            outputs, last_state = tf.nn.static_rnn( cell=cell,
                                                    inputs=inputs, 
                                                    initial_state=self.initial_lstm_state, 
                                                    dtype=tf.float32,
                                                    sequence_length=input_seq_len, 
                                                    scope='encoder')
            outputs_reshape = []
            names = ["enc_mu", "enc_sigma", "enc_mu_post", "enc_sigma_post", "z", "hidden_state"]
            for n,name in enumerate(names):
                x = tf.stack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2]) #now batch*n_steps*n_input
                outputs_reshape.append(x)
            
            enc_mu, enc_sigma, enc_mu_post, enc_sigma_post, z_sample, hidden_state = outputs_reshape
        else:
            outputs, last_state = tf.nn.dynamic_rnn(cell=cell, 
                                                    dtype=tf.float32, 
                                                    sequence_length=input_seq_len,
                                                    inputs=self.input_data, 
                                                    initial_state=self.initial_lstm_state,
                                                    scope="encoder"
                                                    )
            enc_mu, enc_sigma, enc_mu_post, enc_sigma_post, z_sample, hidden_state = outputs


        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma

        self.enc_mu_post = enc_mu_post
        self.enc_sigma_post = enc_sigma_post

        self.final_state_c,self.final_state_h = last_state
        
        #prior from hidden state
        with tf.variable_scope("prior"):
            with tf.variable_scope("hidden"):
                #<hyin/Apr-07-2017> note that h depends on the last z
                prior_hidden = nonlinear(hidden_state, n_units=self.cell.n_prior_hidden, n_layers=1)
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.cell.n_z)
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.exp(linear(prior_hidden, self.cell.n_z, bias_start=0.0))

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.z_sample = z_sample
        self.hidden_state = hidden_state

        if not args.use_cnn:
            with tf.variable_scope("decoder"):
                with tf.variable_scope("hidden"):
                    # dec_hidden = nonlinear(z_sample, n_units=[200, 200, 400, 400], n_layers=4)
                    dec_hidden = nonlinear(tf.concat((z_sample, hidden_state), 2), n_units=[200, 200, 400, 400], n_layers=4)
                    with tf.variable_scope("mu"):
                        dec_mu = tf.nn.sigmoid(linear(dec_hidden, self.cell.n_x))
        else:
            with tf.variable_scope("decoder"):
                dec_hidden = nonlinear(tf.concat((z_sample, hidden_state), 2), n_units=self.cell.n_z, n_layers=1)
                dec_hidden = deconv_decoder(input_=dec_hidden, feature_shape=(28, 28, 1), is_training=self.is_training)
                with tf.variable_scope("mu"):
                    dec_mu = tf.nn.sigmoid(linear(dec_hidden, self.cell.n_x))
        self.mu = dec_mu

        #########
        # for training
        #########
        enc_mu_flatten =  tf.reshape(self.enc_mu_post,[-1, args.latent_size])
        enc_sigma_flatten = tf.reshape(self.enc_sigma_post,[-1, args.latent_size])
        dec_mu_flatten = tf.reshape(self.mu, [-1, args.dim_size])
        prior_mu_flatten = tf.reshape(self.prior_mu, [-1, args.latent_size])
        prior_sigma_flatten = tf.reshape(self.prior_sigma, [-1, args.latent_size])

        flat_target_data = tf.reshape(self.target_data,[-1, args.dim_size])
        self.target = flat_target_data
        self.flat_input = flat_target_data

        self.anneal_rate = tf.Variable(1.0, trainable=False)

        lossfunc = get_lossfunc(enc_mu_flatten, enc_sigma_flatten, dec_mu_flatten, prior_mu_flatten, prior_sigma_flatten, flat_target_data, self.anneal_rate, args)

        with tf.variable_scope('cost'):
            self.cost = lossfunc
        tf.summary.scalar('cost', self.cost)
        # tf.summary.scalar('mu', tf.reduce_mean(self.mu))

        self.merged_summary = tf.summary.merge_all()

        self.lr = tf.Variable(1.00, trainable=False)

        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
        # grads = tf.gradients(self.cost, tvars)
        # grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.lr)

        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.train_op = optimizer.minimize(lossfunc)
        return

    def reconstruct(self, sess, args, seq):
        "a function to test reconstruction especially if the prior is properly learned"

        if self.is_sample:
            prev_state = sess.run(self.cell.zero_state(1, tf.float32))
            # prev_state = self.cell.zero_state(1, tf.float32)
            num = len(seq)

            mus = np.zeros((num, args.dim_size), dtype=np.float32)
            sigmas = np.zeros((num, args.dim_size), dtype=np.float32)

            prior_mus = np.zeros((num, args.latent_size), dtype=np.float32)
            prior_sigmas = np.zeros((num, args.latent_size), dtype=np.float32)
            enc_mus = np.zeros((num, args.latent_size), dtype=np.float32)
            enc_sigmas = np.zeros((num, args.latent_size), dtype=np.float32)

            reconstr_cost = 0

            for i in range(num):
                prev_x = np.zeros((1, 1, args.dim_size), dtype=np.float32)

                prev_x[0][0] = seq[i]

                #<hyin/Jan-23rd-2017> now the problem for the reconstruction is twofold:
                #the second step (the image resulted by the first space hit) is always of a same pattern. And the problem is not the consistency as all the letter images start from
                #a blank image so initially it contains just dot locates at different places. The thing is the pattern is far away from  a dot. But this wrong pattern doesn't seem to harm
                # very much to the following reconstruction
                # the other thing is the if the reconstruction is fed with the synthesized image, the dynamics divert immediately. This sounds like the adversarial samples to NN, where if the input
                # is just slightly corrupted, the output will be drastically impacted. Does that imply an overfitting? Ideally, the latent dynamics should not depend on the input (thus should be robust to corrupted input)
                # but as the comment under the LSTM update indicates, it capture the dynamical behavior poorly... Maybe I should try it again as I fixed the evaluation of next_state_c and next_state_h
                #<hyin/Jul-5th-2017> the step-based reconstruction seems really bad compare with the one processing the entire trajectory. This makes us not able to encode a piece of trajectory. what is the problem?
                if i == 0:
                    feed = {self.input_data: prev_x,
                            self.initial_state_c:prev_state[0],
                            self.initial_state_h:prev_state[1]}

                    # prior_mu, prior_sigma = sess.run([self.enc_mu, self.enc_sigma], feed)
                    prior_mu, prior_sigma = sess.run([self.enc_mu_post, self.enc_sigma_post], feed)
                else:
                    feed = {self.initial_state_c:prev_state[0],
                            self.initial_state_h:prev_state[1]}
                    prior_mu, prior_sigma = sess.run([self.prior_mu, self.prior_sigma], feed)

                feed = {self.input_data: prev_x,
                        self.initial_state_c:prev_state[0],
                        self.initial_state_h:prev_state[1]}


                # this is the working evaluation, what is the difference
                # [next_state_c, next_state_h, o_mu, enc_mu, enc_sigma] = sess.run([self.final_state_c, self.final_state_h, self.mu, self.enc_mu, self.enc_sigma], feed)
                [next_state_c, next_state_h, o_mu, enc_mu, enc_sigma] = sess.run([self.final_state_c, self.final_state_h, self.mu, self.enc_mu_post, self.enc_sigma_post], feed)

                #calculate cost
                input_data = np.zeros((1, 1, args.dim_size), dtype=np.float32)
                # output_data = np.zeros((1, 1, args.dim_size), dtype=np.float32)

                input_data[0][0] = seq[i]
                # output_data[0][0] = mus[i]

                cost = sess.run(self.cost, {self.input_data: input_data, self.target_data: input_data, self.initial_state_c:prev_state[0], self.initial_state_h:prev_state[1]})
                reconstr_cost += cost

                #record all results
                mus[i] = o_mu
                prior_mus[i] = prior_mu
                prior_sigmas[i] = prior_sigma
                enc_mus[i] = enc_mu
                enc_sigmas[i] = enc_sigma

                prev_state = next_state_c, next_state_h

            final_state_c = prev_state[0]
            final_state_h = prev_state[1]
            reconstr_cost = reconstr_cost / float(num)
        else:
            input_data = np.zeros((args.batch_size, args.seq_length, args.dim_size), dtype=np.float32)
            input_data = seq
            feed = {self.input_data: input_data, self.target_data: input_data}
            # reconstr_cost, final_state_c, final_state_h, mus, prior_mus, prior_sigmas, enc_mus, enc_sigmas, flat_input, target = sess.run(
            #         [self.cost, self.final_state_c, self.final_state_h, self.mu, self.prior_mu, self.prior_sigma, self.enc_mu, self.enc_sigma, self.flat_input, self.target], feed)
            reconstr_cost, final_state_c, final_state_h, mus, prior_mus, prior_sigmas, enc_mus, enc_sigmas, flat_input, target = sess.run(
                    [self.cost, self.final_state_c, self.final_state_h, self.mu, self.prior_mu, self.prior_sigma, self.enc_mu_post, self.enc_sigma_post, self.flat_input, self.target], feed)
            mus = np.reshape(mus, (args.batch_size, args.seq_length, args.dim_size))
            # mus = np.rollaxis(mus, 1, 0)

        return mus, prior_mus, prior_sigmas, enc_mus, enc_sigmas, reconstr_cost, final_state_c, final_state_h

    def encode(self, sess, args, seq):
        mus, prior_mus, prior_sigmas, enc_mus, enc_sigmas, reconstr_cost, final_state_c, final_state_h = self.reconstruct(sess, args, seq)
        return prior_mus, prior_sigmas, enc_mus, enc_sigmas, final_state_c, final_state_h

    def sample(self, sess, args, num=20, start=None):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape)*0.0)

        if start is None:
            prev_state = sess.run(self.cell.zero_state(1, tf.float32))
            prev_x = np.zeros((1, 1, args.dim_size))
        else:
            prev_x = np.zeros((1, 1, args.dim_size))
            prev_x[0][0] = start[0]
            prev_state = start[1]

        chunks = np.zeros((num, args.dim_size), dtype=np.float32)
        zs = np.zeros((num, args.latent_size), dtype=np.float32)
        
        for i in xrange(num):
            if i == 0:
                feed = {self.input_data: prev_x,
                        self.initial_state_c:prev_state[0],
                        self.initial_state_h:prev_state[1]}

                # prior_mu, prior_sigma = sess.run([self.enc_mu_post, self.enc_sigma_post], feed)
                prior_mu, prior_sigma = sess.run([self.enc_mu, self.enc_sigma], feed)
            else:
                feed = {self.initial_state_c:prev_state[0],
                        self.initial_state_h:prev_state[1]}
                prior_mu, prior_sigma = sess.run([self.prior_mu, self.prior_sigma], feed)

            # print prior_mu, prior_sigma
            z_sample = prior_mu * 1 + np.random.randn(prev_x.shape[0], args.latent_size) * prior_sigma * 1

            #<hyin/Sep-25th-2017> this does not work, the 'unfetchable' problem is that tensorflow does not allow us to access
            #variable created inside a cond or loop like dynamic_rnn. We might have to hugely restructure the code to move
            #output layers outside the variationalrnn cell
            # enc_mu_post, enc_sigma_post, z_sample = sess.run([self.enc_mu_post, self.enc_sigma_post, self.cell.z], feed)
            o_mu = sess.run([self.mu], {self.hidden_state:[prev_state[1]], self.z_sample:z_sample})[0]
            # print(enc_mu_post, enc_sigma_post, z_sample)

            # curr_x_reshaped = np.zeros((1, 1, args.dim_size), dtype=np.float32)
            # curr_x_reshaped[0][0] = curr_x
            # [next_state_c, next_state_h] = sess.run([self.final_state_c, self.final_state_h], { self.input_data:curr_x_reshaped,
            #                                                                                     self.cell.z:z_sample,
            #                                                                                     self.initial_state_c:prev_state[0],
            #                                                                                     self.initial_state_h:prev_state[1]})

            # <hyin/Jul-14th-2017> today i found that removing x from the state propagation seems to be important if we increase the kl divergence weight, say, to 15
            # but arent the gradients through sequence critical?
            # <hyin/Sep-26th-2017> so cannot be fed again...
            [next_state_c, next_state_h] = sess.run([self.final_state_c, self.final_state_h], {self.cell.z:z_sample[0], self.initial_state_c:prev_state[0], self.initial_state_h:prev_state[1]})


            chunks[i] = o_mu[0][0]
            zs[i] = z_sample
            prev_x = np.zeros((1, 1, args.dim_size), dtype=np.float32)

            prev_x = o_mu
            prev_state = next_state_c, next_state_h

        return chunks, zs

