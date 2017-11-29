import itertools
import json
import logging
import os

import numpy as np
import time
import tensorflow as tf

from utils import update_data, get_data, Logg

logger = logging.getLogger(__name__)


class Model:

    def __init__(self, num_in, parameters, num_out=1):

        self.parameters = parameters

        self.input_ph = tf.placeholder(tf.float32, [None, num_in], name='inputs')
        self.output_ph = tf.placeholder(tf.float32, [None, num_out], name='outputs')

        self.output = self._build(self.input_ph, num_out)

        self.loss = tf.losses.mean_squared_error(self.output_ph, self.output)
        self.update_op = tf.train.AdamOptimizer(parameters.learning_rate).minimize(self.loss)

        self.reset_ops = self._get_reset_op()

    def _build(self, layer, num_out):

        var_names = ['num_units', 'activation_funcs', 'prob_keep']
        iterator = enumerate(zip(*map(self.parameters.__getattribute__, var_names)))

        for i, (num_units, activation_func, prob_keep) in iterator:
            with tf.variable_scope('hidden%i' % i):
                layer = tf.layers.dense(layer, num_units)
                layer = tf.layers.dropout(layer, prob_keep)

        with tf.variable_scope('output'):
            output = tf.layers.dense(layer, num_out, name='output')

        return output

    def _get_reset_op(self):

        # TODO(jalex): Experiment with resetting biases

        reset_ops = []

        for i, prob in enumerate(self.parameters.prob_keep_perm):
            with tf.variable_scope('hidden%i' % i, reuse=True):
                weights = tf.get_variable('dense/kernel')

            mask_reset = tf.cast(tf.random_uniform(weights.shape) < prob, weights.dtype)

            # TODO(jalex): Use initializer? Right now just zeroing them out
            reset_ops.append(tf.assign(weights, weights * mask_reset))

        return reset_ops


def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument('exp_name')
    parser.add_argument('--seed', '-s', type=int, default=0)
    # Model Parameters
    parser.add_argument('--num_layers', '-nl', type=int, default=2)
    parser.add_argument('--num_units', '-nu', nargs='*', type=int, default=16)
    # TODO(jalex): implement changing these in model
    parser.add_argument('--activation_funcs', '-af', nargs='*', default='tanh', choices=['tanh', 'relu'])
    parser.add_argument('--prob_keep', '-pk', nargs='*', default=0.90, type=float)
    # Training parameters
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--num_data_perturbations', '-ndp', type=int, default=10)
    parser.add_argument('--num_epochs', '-ne', type=int, default=1000)
    parser.add_argument('--loss_cutoff', '-lc', type=float, default=0)
    parser.add_argument('--prob_keep_perm', '-pkp', nargs='*', default=0.95, type=float)
    # Debug
    parser.add_argument('--quiet', '-q', action='count', default=0)

    args = parser.parse_args()

    logging.basicConfig(level=args.quiet * 10)

    args.dir_path = os.path.join('.', 'data', args.exp_name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S"))
    args.model_dir_path = os.path.join(args.dir_path, 'models')
    if not os.path.exists(args.model_dir_path):
        os.makedirs(args.model_dir_path)

    attrs_dependent_on_num_layers = ['num_units', 'activation_funcs', 'prob_keep', 'prob_keep_perm']
    for a in attrs_dependent_on_num_layers:
        attr_list = getattr(args, a)
        if not isinstance(attr_list, list):
            attr_list = [attr_list]
        if len(attr_list) == 1:
            setattr(args, a, attr_list * args.num_layers)
        assert len(getattr(args, a)) == args.num_layers

    fpath_params = os.path.join(args.dir_path, 'params.txt')
    with open(fpath_params, 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True))

    return args


def train(parameters, X, logg, model, saver, sess, y):

    for i in range(parameters.num_data_perturbations):

        epoch_counter = range(parameters.num_epochs) if parameters.num_epochs else itertools.count()

        for j in epoch_counter:

            feed = {
                model.input_ph: X,
                model.output_ph: y[:, None],
            }

            loss, _ = sess.run([model.loss, model.update_op], feed)

            logg.record(i, j, loss)

            if j % 100 == 0:
                logger.info('Loss at data perturb %i epoch %i: %f', i, j, loss)

            if loss < parameters.loss_cutoff:
                parameters.loss_cutoff *= 1.03
                logger.info('Num iterations to convergence: %i', j)
                break

        X, y = update_data(X, y)

        saver.save(sess, os.path.join(parameters.model_dir_path, 'model'), global_step=i)

        sess.run(model.reset_ops)


def main(parameters):

    tf.set_random_seed(parameters.seed)
    np.random.seed(parameters.seed)

    logg = Logg(parameters.dir_path)

    X, y = get_data()

    model = Model(X.shape[1], parameters)

    sess = tf.Session().__enter__()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    train(parameters, X, logg, model, saver, sess, y)

    logg.write()


if __name__ == '__main__':

    args = parse_args()

    main(args)
