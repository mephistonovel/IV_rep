try:
    import tensorflow as tf
    import tensorflow.contrib.layers as layers
except:
    pass
import numpy as np
import os
import random
from typing import NamedTuple, Dict, Any, Optional, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class DataSet(NamedTuple):
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray
    
    
class FullDataset(NamedTuple):
    train: DataSet
    valid: DataSet
    test: DataSet

    
def get_var(_dim_in, _dim_out, _name, get_flag=False):
    if get_flag:
        var = tf.get_variable(name=_name, shape=[_dim_in, _dim_out], initializer=tf.contrib.layers.xavier_initializer())
    else:
        var = tf.Variable(tf.random.normal([_dim_in, _dim_out], stddev=0.1 / np.sqrt(_dim_out)), name=_name)
    return var

class AutoIV(object):
    def __init__(self, train_dict, dim_x, dim_v, dim_y):
        """ Build AutoIV model. """

        """ Get sess and placeholder. """
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.dim_x, self.dim_v, self.dim_y = dim_x, dim_v, dim_y
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim_x], name='x')
        self.v = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim_v], name='v')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim_y], name='y')

        """ Set up parameters. """
        self.emb_dim = train_dict['emb_dim']
        self.rep_dim = train_dict['rep_dim']
        self.num = train_dict['num']
        self.coefs = train_dict['coefs']
        self.dropout = train_dict['dropout']
        self.train_flag = tf.compat.v1.placeholder(tf.bool, name='train_flag')
        self.train_dict = train_dict
        self.get_flag = True  # tf.Variable or tf.get_variable
        self.init = tf.contrib.layers.xavier_initializer()

        """ Build model and get loss. """
        self.build_model()
        self.calculate_loss()

    def build_model(self):
        """ Build model. """

        """ Build representation network. """
        with tf.compat.v1.variable_scope('rep'):
            rep_net_layer = self.train_dict['rep_net_layer']
            self.rep_z, self.w_z, self.b_z = self.rep_net(inp=self.v,
                                                          dim_in=self.dim_v,
                                                          dim_out=self.rep_dim,
                                                          layer=rep_net_layer,
                                                          name='rep_z')
            self.rep_c, self.w_c, self.b_c = self.rep_net(inp=self.v,
                                                          dim_in=self.dim_v,
                                                          dim_out=self.rep_dim,
                                                          layer=rep_net_layer,
                                                          name='rep_c')
        self.rep_zc = tf.concat([self.rep_z, self.rep_c], 1)

        """ Build treatment prediction network. """
        with tf.compat.v1.variable_scope('x'):
            self.x_pre, self.w_x, self.b_x = self.x_net(inp=self.rep_zc,
                                                        dim_in=self.rep_dim * 2,
                                                        dim_out=self.dim_x,
                                                        layer=self.train_dict['x_net_layer'])

        """ Build embedding network. """
        with tf.compat.v1.variable_scope('emb'):
            self.x_emb, self.w_emb, self.b_emb = self.emb_net(inp=self.x_pre,
                                                              dim_in=self.dim_x,
                                                              dim_out=self.emb_dim,
                                                              layer=self.train_dict['emb_net_layer'])
        self.rep_cx = tf.concat([self.rep_c, self.x_emb], 1)

        """ Build outcome prediction network. """
        with tf.compat.v1.variable_scope('y'):
            self.y_pre, self.w_y, self.b_y = self.y_net(inp=self.rep_cx,
                                                        dim_in=self.rep_dim + self.emb_dim,
                                                        dim_out=self.dim_y,
                                                        layer=self.train_dict['y_net_layer'])

        """ Maximize MI between z and x. """
        with tf.compat.v1.variable_scope('zx'):
            self.lld_zx, self.bound_zx, self.mu_zx, self.logvar_zx, self.ws_zx = self.mi_net(
                inp=self.rep_z,
                outp=self.x,
                dim_in=self.rep_dim,
                dim_out=self.dim_x,
                mi_min_max='max')

        """ Minimize MI between z and y given x. """
        with tf.compat.v1.variable_scope('zy'):
            self.lld_zy, self.bound_zy, self.mu_zy, self.logvar_zy, self.ws_zy = self.mi_net(
                inp=self.rep_z,
                outp=self.y,
                dim_in=self.rep_dim,
                dim_out=self.dim_y,
                mi_min_max='min',
                name='zy')

        """ Maximize MI between c and x. """
        with tf.compat.v1.variable_scope('cx'):
            self.lld_cx, self.bound_cx, self.mu_cx, self.logvar_cx, self.ws_cx = self.mi_net(
                self.rep_c,
                outp=self.x,
                dim_in=self.rep_dim,
                dim_out=self.dim_x,
                mi_min_max='max')

        """ Maximize MI between c and y. """
        with tf.compat.v1.variable_scope('cy'):
            self.lld_cy, self.bound_cy, self.mu_cy, self.logvar_cy, self.ws_cy = self.mi_net(
                inp=self.rep_c,
                outp=self.y,
                dim_in=self.rep_dim,
                dim_out=self.dim_y,
                mi_min_max='max')

        """ Minimize MI between z and c. """
        with tf.compat.v1.variable_scope('zc'):
            self.lld_zc, self.bound_zc, self.mu_zc, self.logvar_zc, self.ws_zc = self.mi_net(
                inp=self.rep_z,
                outp=self.rep_c,
                dim_in=self.rep_dim,
                dim_out=self.rep_dim,
                mi_min_max='min')

    def calculate_loss(self):
        """ Get loss."""

        """ Loss of y prediction. """
        self.loss_cx2y = tf.reduce_mean(tf.square(self.y - self.y_pre))

        """ Loss of t prediction. """
        self.loss_zc2x = tf.reduce_mean(tf.square(self.x - self.x_pre))

        """ Loss of network regularization. """
        def w_reg(w):
            """ Calculate l2 loss of network weight. """
            w_reg_sum = 0
            for w_i in range(len(w)):
                w_reg_sum = w_reg_sum + tf.nn.l2_loss(w[w_i])
            return w_reg_sum
        self.loss_reg = (w_reg(self.w_z) + w_reg(self.w_c) +
                         w_reg(self.w_emb) + w_reg(self.w_x) + w_reg(self.w_y)) / 5.

        """ Losses. """
        self.loss_lld = self.coefs['coef_lld_zy'] * self.lld_zy + \
                        self.coefs['coef_lld_cx'] * self.lld_cx + \
                        self.coefs['coef_lld_zx'] * self.lld_zx + \
                        self.coefs['coef_lld_cy'] * self.lld_cy + \
                        self.coefs['coef_lld_zc'] * self.lld_zc

        self.loss_bound = self.coefs['coef_bound_zy'] * self.bound_zy + \
                          self.coefs['coef_bound_cx'] * self.bound_cx + \
                          self.coefs['coef_bound_zx'] * self.bound_zx + \
                          self.coefs['coef_bound_cy'] * self.bound_cy + \
                          self.coefs['coef_bound_zc'] * self.bound_zc + \
                          self.coefs['coef_reg'] * self.loss_reg

        self.loss_2stage = self.coefs['coef_cx2y'] * self.loss_cx2y + \
                           self.coefs['coef_zc2x'] * self.loss_zc2x + \
                           self.coefs['coef_reg'] * self.loss_reg

    def layer_out(self, inp, w, b, flag):
        """ Set up activation function and dropout for layers."""
        out = tf.matmul(inp, w) + b
        if flag:
            return tf.layers.dropout(tf.nn.elu(out), rate=self.dropout, training=self.train_flag)
        else:
            return out

    def rep_net(self, inp, dim_in, dim_out, layer, name):
        """ Representation network. """
        rep, w_, b_ = [inp], [], []
        with tf.compat.v1.variable_scope(name):
            for i in range(layer):
                dim_in_net = dim_in if (i == 0) else dim_out
                dim_out_net = dim_out
                w_.append(get_var(dim_in_net, dim_out_net, 'w_' + name + '_%d' % i, get_flag=self.get_flag))
                b_.append(tf.Variable(tf.zeros([1, dim_out_net]), name='b_' + name + '_%d' % i))
                rep.append(self.layer_out(rep[i], w_[i], b_[i], flag=(i != layer - 1)))
        return rep[-1], w_, b_

    def x_net(self, inp, dim_in, dim_out, layer):
        """ Treatment prediction network. """
        x_pre, w_x, b_x = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_in // (i * 2)
            dim_out_net = dim_in // ((i + 1) * 2) if i != (layer - 1) else dim_out
            dim_in_net = dim_in_net if dim_in_net > 0 else 1
            dim_out_net = dim_out_net if dim_out_net > 0 else 1
            w_x.append(get_var(dim_in_net, dim_out_net, 'w_x' + '_%d' % i, get_flag=self.get_flag))
            b_x.append(tf.Variable(tf.zeros([1, dim_out_net]), name='b_x' + '_%d' % i))
            x_pre.append(self.layer_out(x_pre[i], w_x[i], b_x[i], flag=(i != layer - 1)))
        return x_pre[-1], w_x, b_x

    def emb_net(self, inp, dim_in, dim_out, layer):
        """ Treatment embedding network. """
        x_emb, w_emb, b_emb = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_out
            dim_out_net = dim_out
            w_emb.append(get_var(dim_in_net, dim_out_net, 'w_emb_%d' % i, get_flag=self.get_flag))
            b_emb.append(tf.Variable(tf.zeros([1, dim_out_net]), name='b_emb_%d' % i))
            x_emb.append(self.layer_out(x_emb[i], w_emb[i], b_emb[i], flag=(i != layer - 1)))
        return x_emb[-1], w_emb, b_emb

    def y_net(self, inp, dim_in, dim_out, layer):
        """ Outcome prediction network. """
        y_pre, w_y, b_y = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_in // (i * 2)
            dim_out_net = dim_in // ((i + 1) * 2) if i != (layer - 1) else dim_out
            dim_in_net = dim_in_net if dim_in_net > 0 else 1
            dim_out_net = dim_out_net if dim_out_net > 0 else 1
            w_y.append(get_var(dim_in_net, dim_out_net, 'w_y' + '_%d' % i, get_flag=self.get_flag))
            b_y.append(tf.Variable(tf.zeros([1, dim_out_net]), name='b_y' + '_%d' % i))
            y_pre.append(self.layer_out(y_pre[i], w_y[i], b_y[i], flag=(i != layer - 1)))
        return y_pre[-1], w_y, b_y

    def fc_net(self, inp, dim_out, act_fun, init):
        """ Fully-connected network. """
        return layers.fully_connected(inputs=inp,
                                      num_outputs=dim_out,
                                      activation_fn=act_fun,
                                      weights_initializer=init)

    def mi_net(self, inp, outp, dim_in, dim_out, mi_min_max, name=None):
        """ Mutual information network. """
        h_mu = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        mu = self.fc_net(h_mu, dim_out, None, self.init)
        h_var = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        logvar = self.fc_net(h_var, dim_out, tf.nn.tanh, self.init)

        new_order = tf.random_shuffle(tf.range(self.num))
        outp_rand = tf.gather(outp, new_order)

        """ Get likelihood. """
        loglikeli = -tf.reduce_mean(tf.reduce_sum(-(outp - mu) ** 2 / tf.exp(logvar) - logvar, axis=-1))

        """ Get positive and negative U."""
        pos = - (mu - outp) ** 2 / tf.exp(logvar)
        neg = - (mu - outp_rand) ** 2 / tf.exp(logvar)

        if name == 'zy':
            x_rand = tf.gather(self.x, new_order)

            # Using RBF kernel to measure distance.
            sigma = self.train_dict['sigma']
            w = tf.exp(-tf.square(self.x - x_rand) / (2 * sigma ** 2))
            w_soft = tf.nn.softmax(w, axis=0)
        else:
            w_soft = 1. / self.num

        """ Get estimation of mutual information. """
        if mi_min_max == 'min':
            pn = 1.
        elif mi_min_max == 'max':
            pn = -1.
        else:
            raise ValueError
        bound = pn * tf.reduce_sum(w_soft * (pos - neg))

        return loglikeli, bound, mu, logvar, w_soft



def get_tf_var(names):
    _vars = []
    for na_i in range(len(names)):
        _vars = _vars + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
    return _vars

def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(lrate, global_step, NUM_ITER_PER_DECAY, lrate_decay, staircase=True)
    opt = tf.compat.v1.train.AdamOptimizer(lr)
    train_opt = opt.minimize(loss, global_step=global_step, var_list=_vars)
    return train_opt

def Auto_IV(train_data_full,test_data_full, i,args):
    # data.numpy()
    ### data preprocessing ### 
    x, t, y = train_data_full
    (x_test, t_test, y_test) = test_data_full
    
    y = torch.from_numpy(y.cpu()) if isinstance(y, np.ndarray) else y.cpu()
    t = torch.from_numpy(t.cpu()) if isinstance(t, np.ndarray) else t.cpu()
    x = torch.from_numpy(x).cpu() if isinstance(x, np.ndarray) else x.cpu()
    
    y_test = torch.from_numpy(y_test.cpu()) if isinstance(y_test, np.ndarray) else y_test.cpu()
    t_test = torch.from_numpy(t_test.cpu()) if isinstance(t_test, np.ndarray) else t_test.cpu()
    x_test = torch.from_numpy(x_test.cpu()) if isinstance(x_test, np.ndarray) else x_test.cpu()
    

    prep_data = np.concatenate([x,t.reshape(-1,1), y.reshape(-1,1)],axis=1)
    prep_data_test = np.concatenate([x_test,t_test.reshape(-1,1), y_test.reshape(-1,1)],axis=1)

    
    infer_input_train = DataSet(t=prep_data[:,-2].reshape(-1,1),
                          x=prep_data[:,:-2],
                          y=prep_data[:,-1].reshape(-1,1),
                          v=prep_data[:,-1].reshape(-1,1))
    
    infer_input_test = DataSet(t=prep_data_test[:,-2].reshape(-1,1),
                          x=prep_data_test[:,:-2],
                          y=prep_data_test[:,-1].reshape(-1,1),
                          v=prep_data_test[:,-1].reshape(-1,1))
    
    np.random.seed(i)
    np.random.shuffle(prep_data)
    
    train_ratio = 0.8
    validation_ratio = 0.2
    # test_ratio = 0.2

    total_samples = len(prep_data)
    train_split = int(total_samples * train_ratio)
    validation_split = int(total_samples * (train_ratio + validation_ratio))

    train_data = prep_data[:train_split]
    valid_data = prep_data[train_split:]
    # test_data = prep_data[validation_split:]
    
    train_tup = DataSet(t= train_data[:,-2].reshape(-1,1),
                        x= train_data[:,:-2],
                        y = train_data[:,-1].reshape(-1,1),
                        v=train_data[:,-1].reshape(-1,1))

    valid_tup = DataSet(t= valid_data[:,-2].reshape(-1,1),
                        x= valid_data[:,:-2],
                        y = valid_data[:,-1].reshape(-1,1),
                        v = valid_data[:,-1].reshape(-1,1))

    test_tup = DataSet(t= infer_input_test.t,
                        x= infer_input_test.x,
                        y = infer_input_test.y,
                        v = infer_input_test.v)
    
    data = FullDataset(train= train_tup,
                       valid= valid_tup,
                       test= test_tup)
    
    ### train dict ### 
    train_dict = {}
    
    train_dict['seed'] = i
    train_dict['emb_dim'] = 1
    train_dict['rep_dim'] = 1
    train_dict['coefs'] = {'coef_cx2y': 1, 'coef_zc2x': 1, 'coef_lld_zx': 1,
                'coef_lld_zy': 1, 'coef_lld_cx': 1,  'coef_lld_cy': 1,
                'coef_lld_zc': 1, 'coef_bound_zx': 1, 'coef_bound_zy': 1,
                'coef_bound_cx': 1, 'coef_bound_cy': 1, 'coef_bound_zc': 1, 'coef_reg': 0.001}
    train_dict['dropout'] = 0.0
    train_dict['rep_net_layer'] = 2
    train_dict['x_net_layer'] = 2
    train_dict['emb_net_layer'] = 2
    train_dict['y_net_layer'] = 2
    train_dict['sigma'] = 0.1
    train_dict['lrate'] = 1e-3
    train_dict['opt_lld_step'] = 1
    train_dict['opt_bound_step'] = 1
    train_dict['opt_2stage_step'] = 1
    train_dict['epochs'] = 1000
    train_dict['interval'] = 10
    train_dict['num'] = train_split

    tf.reset_default_graph()
    random.seed(train_dict['seed'])
    tf.compat.v1.set_random_seed(train_dict['seed'])
    # np.random.seed(train_dict['seed'])
    os.environ['PYTHONHASHSEED'] = str(train_dict['seed'])

    tf.compat.v1.reset_default_graph()
    dim_x, dim_v, dim_y = data.train.t.shape[1], data.train.x.shape[1], data.train.y.shape[1]
    model = AutoIV(train_dict, dim_x, dim_v, dim_y)

    """ Get trainable variables. """
    zx_vars = get_tf_var(['zx'])
    zy_vars = get_tf_var(['zy'])
    cx_vars = get_tf_var(['cx'])
    cy_vars = get_tf_var(['cy'])
    zc_vars = get_tf_var(['zc'])
    rep_vars = get_tf_var(['rep/rep_z', 'rep/rep_c'])
    x_vars = get_tf_var(['x'])
    emb_vars = get_tf_var(['emb'])
    y_vars = get_tf_var(['y'])

    vars_lld = zx_vars + zy_vars + cx_vars + cy_vars + zc_vars
    vars_bound = rep_vars
    vars_2stage = rep_vars + x_vars + emb_vars + y_vars

    """ Set optimizer. """
    train_opt_lld = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
                            lrate_decay=0.95, loss=model.loss_lld, _vars=vars_lld)

    train_opt_bound = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
                                lrate_decay=0.95, loss=model.loss_bound, _vars=vars_bound)

    train_opt_2stage = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
                                lrate_decay=0.95, loss=model.loss_2stage, _vars=vars_2stage)

    train_opts = [train_opt_lld, train_opt_bound, train_opt_2stage]
    train_steps = [train_dict['opt_lld_step'], train_dict['opt_bound_step'], train_dict['opt_2stage_step']]

    # model, train_opts, train_steps, data.train
    # Begin Train
    model.sess.run(tf.compat.v1.global_variables_initializer())

    """ Training, validation, and test dict. """
    dict_train_true = {model.v: data.train.x, model.x: data.train.t, model.y: data.train.y, model.train_flag: True}
    dict_train = {model.v: data.train.x, model.x: data.train.t, model.x_pre: data.train.t, model.y: data.train.y, model.train_flag: False}
    dict_valid = {model.v: data.valid.x, model.x: data.valid.t, model.x_pre: data.valid.t, model.y: data.valid.y, model.train_flag: False}
    dict_test = {model.v: data.test.x, model.x_pre: data.test.t, model.y: data.test.y, model.train_flag: False}

    epochs = train_dict['epochs']
    intt = train_dict['epochs'] // train_dict['interval']
    for ep_th in range(epochs):
        if (ep_th % intt == 0) or (ep_th == epochs - 1):
            loss = model.sess.run([model.loss_cx2y,
                                    model.loss_zc2x,
                                    model.lld_zx,
                                    model.lld_zy,
                                    model.lld_cx,
                                    model.lld_cy,
                                    model.lld_zc,
                                    model.bound_zx,
                                    model.bound_zy,
                                    model.bound_cx,
                                    model.bound_cy,
                                    model.bound_zc,
                                    model.loss_reg],
                                    feed_dict=dict_train)
            y_pre_train = model.sess.run(model.y_pre, feed_dict=dict_train)
            y_pre_valid = model.sess.run(model.y_pre, feed_dict=dict_valid)
            y_pre_test = model.sess.run(model.y_pre, feed_dict=dict_test)

            mse_train = np.mean(np.square(y_pre_train - data.train.v))
            mse_valid = np.mean(np.square(y_pre_valid - data.valid.v))
            mse_test = np.mean(np.square(y_pre_test - data.test.v))

            print("Epoch {}: {} - {} - {}".format(ep_th, mse_train, mse_valid, mse_test))
        for i in range(len(train_opts)):  # optimizer to train
            for j in range(train_steps[i]):  # steps of optimizer
                model.sess.run(train_opts[i], feed_dict=dict_train_true)

    def get_rep_z(data):
        dict_data = {model.v: data.x, model.x_pre: data.t, model.y: data.y, model.train_flag: False}
        data_z = model.sess.run(model.rep_z, feed_dict=dict_data)
        return data_z

    rep_z_train = get_rep_z(infer_input_train)
    rep_z_test = get_rep_z(infer_input_test)
    # data.train.z = rep_z
    
    return rep_z_train,rep_z_test