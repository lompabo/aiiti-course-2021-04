#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

# Configuration
figsize = (9, 3)

def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=figsize, autoclose=True, s=4):
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(labels))
    plt.tight_layout()


def plot_series(series, labels=None,
        figsize=figsize, autoclose=True, s=4):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(series.index, series, label='data')
    if labels is not None:
        # nonzero = series.index[labels != 0]
        smin, smax = np.min(series),  np.max(series)
        lvl = smin - 0.05 * (smax-smin)
        # plt.scatter(nonzero, np.ones(len(nonzero)) * lvl,
        #         s=s,
        #         color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(labels))
    plt.tight_layout()


def plot_stoch_series(data, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    means = np.mean(data.values, axis=1)
    stds = np.std(data.values, axis=1)
    plt.plot(data.index, means, label='mean')
    plt.fill_between(data.index,
            means-stds,
            means+stds,
            alpha=0.1,
            label='min-max')
    plt.tight_layout()


def plot_signal(signal, labels=None,
        figsize=figsize, autoclose=True, s=4):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(signal.index, signal, label='signal')
    if labels is not None:
        nonzero = signal.index[labels != 0]
        smin, smax = np.min(signal),  np.max(signal)
        lvl = smin - 0.05 * (smax-smin)
        plt.scatter(nonzero, np.ones(len(nonzero)) * lvl,
                s=s, color='tab:orange')
    plt.tight_layout()


def plot_bars(data, figsize=figsize, autoclose=True, tick_gap=1):
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    x = 0.5 + np.arange(len(data))
    plt.bar(x, data, width=0.7)
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.tight_layout()


def plot_training_history(history, 
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], label='val. loss')
        plt.legend()
    plt.tight_layout()


def plot_distribution_2D(estimator=None, samples=None,
        xr=None, yr=None,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if samples is not None:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
    if estimator is not None:
        if xr is None:
            xr = np.linspace(-1, 1, 100)
        if yr is None:
            yr = np.linspace(-1, 1, 100)
        nx = len(xr)
        ny = len(yr)
        xc = np.repeat(xr, ny)
        yc = np.tile(yr, nx)
        data = np.vstack((xc, yc)).T
        dvals = np.exp(estimator.score_samples(data))
        dvals = dvals.reshape((nx, ny))
        plt.imshow(dvals.T[::-1, :], aspect='auto')
        plt.xticks(np.linspace(0, len(xr), 5), np.linspace(xr[0], xr[-1], 5))
        plt.xticks(np.linspace(0, len(yr), 5), np.linspace(yr[0], yr[-1], 5))
    plt.tight_layout()


def get_errors(signal, labels, thr, tolerance=1):
    pred = signal[signal > thr].index
    anomalies = labels[labels != 0].index

    fp = set(pred)
    fn = set(anomalies)
    for lag in range(-tolerance, tolerance+1):
        fp = fp - set(anomalies+lag)
        fn = fn - set(pred+lag)
    return fp, fn


class HPCMetrics:
    def __init__(self, c_alarm, c_missed, tolerance):
        self.c_alarm = c_alarm
        self.c_missed = c_missed
        self.tolerance = tolerance

    def cost(self, signal, labels, thr):
        # Obtain errors
        fp, fn = get_errors(signal, labels, thr, self.tolerance)

        # Compute the cost
        return self.c_alarm * len(fp) + self.c_missed * len(fn)


def opt_threshold(signal, labels, th_range, cmodel):
    costs = [cmodel.cost(signal, labels, th) for th in th_range]
    best_th = th_range[np.argmin(costs)]
    best_cost = np.min(costs)
    return best_th, best_cost


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def collect_training(data, tr_ts_ratio):
    if isinstance(data, dict):
        tr_list = []
        for key, kdata in data.items():
            sep = int(np.round(tr_ts_ratio * len(kdata)))
            tr_list.append(kdata.iloc[:sep])
        tr_data = pd.concat(tr_list)
    else:
        sep = int(np.round(tr_ts_ratio * len(data)))
        tr_data = data.iloc[:sep]
    return tr_data


def kde_ad(x_tr, x_vs, y_vs, x_ts, y_ts, th_range, h_range, cmodel):
    # Optimize the bandwidth, if more than one value is given
    if len(h_range) > 1:
        params = {'bandwidth': h_range}
        opt = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
        opt.fit(x_tr)
        h = opt.best_params_['bandwidth']
    else:
        h = h_range[0]
    # Traing a KDE estimator
    kde = KernelDensity(bandwidth=h)
    kde.fit(x_tr)
    # Generate a signal for the validation set
    ldens_vs = kde.score_samples(x_vs)
    signal_vs = pd.Series(index=x_vs.index, data=-ldens_vs)
    # Threshold optimization
    th, val_cost = opt_threshold(signal_vs, y_vs, th_range, cmodel)
    # Compute the cost over the test set
    ldens_ts = kde.score_samples(x_ts)
    signal_ts = pd.Series(index=x_ts.index, data=-ldens_ts)
    ts_cost = cmodel.cost(signal_ts, y_ts, th)
    # Return results
    return kde, ts_cost


def ae_ad(ae, x_tr, x_vs, y_vs, x_ts, y_ts, th_range, cmodel):
    # Traing the autoencoder
    ae.compile(optimizer='RMSProp', loss='mse')
    cb = [callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    ae.fit(x_tr, x_tr, validation_split=0.1, callbacks=cb,
           batch_size=32, epochs=20, verbose=0)
    # Generate a signal for the validation set
    preds_vs = pd.DataFrame(index=x_vs.index,
            columns=x_vs.columns, data=ae.predict(x_vs))
    sse_vs = np.sum(np.square(preds_vs - x_vs), axis=1)
    signal_vs = pd.Series(index=x_vs.index, data=sse_vs)
    # Threshold optimization
    th, val_cost = opt_threshold(signal_vs, y_vs, th_range, cmodel)
    # Compute the cost over the test set
    preds_ts = pd.DataFrame(index=x_ts.index,
            columns=x_ts.columns, data=ae.predict(x_ts))
    sse_ts = np.sum(np.square(preds_ts - x_ts), axis=1)
    signal_ts = pd.Series(index=x_ts.index, data=sse_ts)
    ts_cost = cmodel.cost(signal_ts, y_ts, th)
    # Return results
    return ts_cost

def coupling(input_shape, nunits=64, nhidden=2, reg=0.01):
    assert(nhidden >= 0)    
    x = keras.layers.Input(shape=input_shape)
    # Build the layers for the t transformation (translation)
    t = x
    for i in range(nhidden):
        t = Dense(nunits, activation="relu", kernel_regularizer=l2(reg))(t)
    t = Dense(input_shape, activation="linear", kernel_regularizer=l2(reg))(t)
    # Build the layers for the s transformation (scale)
    s = x
    for i in range(nhidden):
        s = Dense(nunits, activation="relu", kernel_regularizer=l2(reg))(s)
    s = Dense(input_shape, activation="tanh", kernel_regularizer=l2(reg))(s)
    # Return the layers, wrapped in a keras Model object
    return keras.Model(inputs=x, outputs=[s, t])


class RealNVP(keras.Model):
    def __init__(self, input_shape, num_coupling, units_coupling=32, depth_coupling=0,
            reg_coupling=0.01):
        super(RealNVP, self).__init__()
        self.num_coupling = num_coupling
        # Distribution of the latent space
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(input_shape, dtype=np.float32),
            scale_diag=np.ones(input_shape, dtype=np.float32)
        )
        # Build a mask
        half_n = int(np.ceil(input_shape/2))
        m1 = ([0, 1] * half_n)[:input_shape]
        m2 = ([1, 0] * half_n)[:input_shape]
        self.masks = np.array([m1, m2] * (num_coupling // 2), dtype=np.float32)
        # Choose what to track at training time
        self.loss_tracker = keras.metrics.Mean(name="loss")
        #  Build layers
        self.layers_list = [coupling(input_shape, units_coupling, depth_coupling, reg_coupling)
                            for i in range(num_coupling)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv, direction = 0, 1
        if training: direction = -1
        for i in range(self.num_coupling)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s, t = s*reversed_mask, t*reversed_mask
            gate = (direction - 1) / 2
            x = reversed_mask * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s)) \
                + x_masked
            log_det_inv += gate * tf.reduce_sum(s, axis=1)
        return x, log_det_inv

    def log_loss(self, x):
        log_densities = self.score_samples(x)
        return -tf.reduce_mean(log_densities)

    def score_samples(self, x):
        y, logdet = self(x)
        log_probs = self.distribution.log_prob(y) + logdet
        return log_probs

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}



def plot_rnvp_transformation(rnvp, xr=None, yr=None,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    # Define ranges
    if xr is None:
        xr = np.linspace(-1, 1, 7, dtype=np.float32)
    if yr is None:
        yr = np.linspace(-1, 1, 7, dtype=np.float32)
    # Build the input set
    nx = len(xr)
    ny = len(yr)
    xc = np.repeat(xr, ny)
    yc = np.tile(yr, nx)
    data = np.vstack((xc, yc)).T
    # Transform the input step
    z, _ = rnvp(data, training=False)
    # Obtain traces
    traces = np.concatenate((
        data.reshape(1, -1, 2),
        z.numpy().reshape(1, -1, 2),
        ))
    # Plot traces
    for i in range(traces.shape[1]):
        plt.plot(traces[:, i, 0], traces[:, i, 1], ':',
                color='0.8', zorder=0)
        xh = plt.scatter(data[:, 0], data[:, 1],
                color='tab:blue', s=4, zorder=1)
        zh = plt.scatter(z.numpy()[:, 0], z.numpy()[:, 1],
                color='tab:orange', s=4, zorder=1)
    plt.legend([xh, zh], ['x', 'z'])
    plt.tight_layout()


