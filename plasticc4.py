# Import Packages
####################################################################
import numpy as np, pandas as pd, tensorflow as tf, xgboost as xgb
from random import shuffle
from tqdm import tqdm
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
import re, pprint, re, string, sklearn, gc, time
from sklearn.model_selection import train_test_split
from functools import reduce
from datetime import datetime
import sklearn.preprocessing
from operator import itemgetter
from sklearn.preprocessing import StandardScaler

# Import Data
####################################################################
folder_path = "C:/Users/user/Desktop/plasticc/"
save_folder_path = "C:/Users/user/Desktop/plasticc/tf_model_dir/"
train = pd.read_csv(folder_path + "training_set.csv")
train_meta = pd.read_csv(folder_path + "training_set_metadata.csv")
test = pd.read_csv(folder_path + "test_reform.csv")
test_pred_path = "C:/Users/user/Desktop/plasticc/test_output/"

# Define Functions
####################################################################
def slice_by_index(lst, indices):
    """slice a list with a list of indices"""
    slicer = itemgetter(*indices)(lst)
    if len(indices) == 1:
        return [slicer]
    return list(slicer)

def shuffle_batch(y, batch_size):
    rnd_idx = np.random.permutation(len(y))
    n_batches = len(y) // batch_size
    batch_list = []
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_list.append([z for z in batch_idx])
    return batch_list

def seconds_to_time(sec):
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def mjd_to_unix(df, mjd_col):
    temp_list = []
    for mjd in df[mjd_col]:
        temp_list.append((mjd - 40587) * 86400)
    return temp_list

def plastic_ts_agg(dat, metadat):
    df_copy = dat.\
    groupby(['object_id', 'mjd', 'passband'], axis = 0, as_index = False).\
    agg({'flux': [np.min, np.max, np.mean],
         'flux_err': [np.min, np.max, np.mean],
         'detected': [np.max]}).\
    sort_values(['object_id', 'mjd', 'passband'], axis = 0).\
    fillna(0)
    df_copy.columns = ['object_id', 'mjd', 'passband', 'min_flux', 'max_flux', 
                       'mean_flux', 'min_flux_err', 'max_flux_err', 'mean_flux_err', 'max_detected']
    df_copy['passband'] = df_copy['passband'].astype(int)
    df_copy['object_id'] = df_copy['object_id'].astype(int)
    output = df_copy.sort_values(['object_id', 'mjd', 'passband'])
    start_tm = output[['object_id', 'mjd']]
    start_tm.columns = ['object_id', 'mjd_start']
    start_tm = start_tm.\
    groupby(['object_id'], as_index = False).\
    agg({'mjd_start':'min'})
    output = pd.merge(output, start_tm, 'left', 'object_id')
    output['unix_mjd'] = mjd_to_unix(output, 'mjd')
    output['unix_mjd_start'] = mjd_to_unix(output, 'mjd_start')
    output['tm_elapsed'] = [np.float64(i) for i in output['unix_mjd'] - output['unix_mjd_start']]
    output['dt'] = [datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S') for i in output['unix_mjd']]
    output['doy'] = [int(datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime('%j')) for i in output['dt']]
    output.drop(['mjd_start', 'unix_mjd', 'unix_mjd_start', 'dt', 'mjd'], inplace = True, axis = 1)
    output = pd.merge(output, metadat, 'inner', 'object_id').fillna(0)
    return output

def pd_to_array_bycol(df, bycol, ycol):
    x_list = []
    y_list = []
    uniq_colvals = [ucv for ucv in set(df[bycol])]
    for ucv in tqdm(uniq_colvals):
        x_list.append(df[df[bycol] == ucv].drop([bycol, ycol], axis = 1).values.astype('float32'))
        y_list.append(int(df[df[bycol] == ucv].iloc[0,:][ycol]))
    return np.asarray(y_list), np.array(x_list)

def pd_to_array_bycol_test(df, bycol, ycol):
    x_list = []
    uniq_colvals = [ucv for ucv in set(df[bycol])]
    x_cols = [c for c in df.columns if c not in [bycol, ycol]]
    for ucv in tqdm(uniq_colvals):
        x_list.append(df[df[bycol] == ucv][x_cols].values.astype('float32'))
    return np.array(x_list)

def array_resizing(arrays):
    nrow_list = [i.shape[0] for i in arrays]
    max_row_size = max(nrow_list)
    ncol = arrays[0].shape[1]
    temp_list = []
    for arr in arrays:
        num_add_rows = max_row_size - arr.shape[0]
        add_rows = np.zeros((num_add_rows, ncol))
        concat_rows = np.concatenate([add_rows,arr])
        temp_list.append(concat_rows)
    return np.asarray(temp_list)

def replace_with_dict(ar, dic):
    keys = np.array(list(dic.keys()))
    vals = np.array(list(dic.values()))
    ord_keys = keys.argsort()
    return vals[ord_keys[np.searchsorted(keys, ar, sorter = ord_keys)]]

def inverse_class_weights(y):
    uniq_y = sorted(set(y))
    temp_list = []
    for uy in uniq_y:
        temp_list.append(1 - (np.sum([1 if i == uy else 0 for i in y]) / len(y)))
    return temp_list

def multiclass_rnn(trn_x, val_x, tst_x, trn_y, val_y, inv_class_wt, n_epochs, neurons, early_stop, lrate, l1_reg, batch_size, save_dir):
    start_tm = time.time()
    tf.reset_default_graph()
    # Placeholders
    X = tf.placeholder(tf.float32, [None, None, trn_x.shape[2]])
    y = tf.placeholder(tf.int32, [None])
    # Layer Topology
    rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = neurons, activation = tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype = tf.float32)
    logits = tf.layers.dense(states, len(set(trn_y)), kernel_regularizer = tf.contrib.layers.l1_regularizer(l1_reg))
    # Loss Function
    class_weights = tf.constant(inv_class_wt)
    weights = tf.gather(class_weights, y)
    xentropy = tf.losses.sparse_softmax_cross_entropy(labels = y,logits = logits, weights = weights)                        
    loss = tf.reduce_mean(xentropy)
    # Optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lrate)
    training_op = optimizer.minimize(loss)
    # Evaluation
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    pred_prob = tf.nn.softmax(logits, name = "pred_prob")
    # Training
    batch_i = shuffle_batch(y = trn_y, batch_size = batch_size)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        best_loss = 99999999
        trn_loss_list = []
        val_loss_list = []
        epoch_list = []
        break_list = []
        for epoch in range(n_epochs):
            for i in range(trn_x.shape[0] // batch_size):
                y_batch = trn_y[batch_i[i]]
                x_batch = trn_x[batch_i[i]]
                sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            train_loss = loss.eval(feed_dict={X: trn_x, y: trn_y})
            val_loss = loss.eval(feed_dict={X: valid_x_resized, y: val_y})
            print(str(int(epoch + 1)), "Train Loss:", train_loss, "-------","Val Loss:", val_loss)
            val_loss_list.append(val_loss)
            trn_loss_list.append(train_loss)
            epoch_list.append(int(epoch + 1))
            best_loss = min(val_loss_list)
            if val_loss > best_loss:
                break_list.append(1)
            else:
                break_list = []
                saver.save(sess, save_folder_path)
            if sum(break_list) >= early_stop:
                print("Stopping after " + str(int(epoch + 1)) + " epochs because validation hasn't improved in " + str(early_stop) + " round(s).")
                break
    print("Elapsed Time: " + seconds_to_time(np.float64(time.time() - start_tm)))
    print("Model saved in path: " + save_folder_path)
    # Output - if test set provided = predicted probabilities
    #        - else dataframe showing training progress 
    if tst_x is not None:
        temp_list = []
        with tf.Session() as sess:
            saver.restore(sess, save_folder_path)
            pred_prob_arr = pred_prob.eval(feed_dict={X: trn_x})
            for i in range(pred_prob_arr.shape[0]):
                temp_list.append([j for j in pred_prob_arr[i,:]])
            output_df = pd.DataFrame(temp_list, columns = ['class_' + str(i+1) for i in range(len(set(trn_y)))])
        return output_df
    else:
        training_prog = pd.DataFrame({'Epoch': epoch_list, 'Training_Loss': trn_loss_list, 'Validation_Loss': val_loss_list})
        return training_prog

# Execute Data Prep Functions (Training)
####################################################################
# Split Train and Validation
train_xy = plastic_ts_agg(train, train_meta)
train_xy, valid_xy = train_test_split(train_xy, test_size = 0.2, random_state = 11062018)
del train; del train_meta; gc.collect()

# Scale Data
x_cols = [c for c in train_xy.columns if c not in ['target', 'object_id']]
scaler = StandardScaler()
temp_train_x = pd.DataFrame(scaler.fit_transform(train_xy[x_cols]),
                            index = train_xy.index,
                            columns = x_cols)
temp_valid_x = pd.DataFrame(scaler.transform(valid_xy[x_cols]),
                            index = valid_xy.index,
                            columns = x_cols)
temp_train_y = train_xy[['target', 'object_id']]
temp_valid_y = valid_xy[['target', 'object_id']]
train_xy = pd.concat([temp_train_y, temp_train_x], axis = 1)
valid_xy = pd.concat([temp_valid_y, temp_valid_x], axis = 1)
del temp_train_x; del temp_valid_x; del temp_valid_y; del temp_train_y; gc.collect()
temp_test_x = pd.DataFrame(scaler.transform(test[x_cols]),
                      index = test.index,
                      columns = x_cols)
temp_test_id = test[['object_id']]
test_xy = pd.concat([temp_test_id, temp_test_x], axis = 1)
del temp_test_x; del temp_test_id; gc.collect()
test_id = test[['object_id']]

# Split X and Y, Resize X, Replace Target Values
train_y, train_x = pd_to_array_bycol(train_xy, 'object_id', 'target')
valid_y, valid_x = pd_to_array_bycol(valid_xy, 'object_id', 'target')
test_x = pd_to_array_bycol_test(test, 'object_id', 'target')
train_x_resized = array_resizing(train_x)
valid_x_resized = array_resizing(valid_x)
test_x_resized = array_resizing(test_x)
class_lookup = {6:0, 15:1, 16:2, 42:3, 52:4, 53:5, 62:6, 64:7, 65:8, 67:9, 88:10, 90:11, 92:12, 95:13}
train_y = replace_with_dict(np.array(train_y), class_lookup)
valid_y = replace_with_dict(np.array(valid_y), class_lookup)
del train_xy; del valid_xy; gc.collect()

# Class Weights
####################################################################
class_reweighting = inverse_class_weights(train_y)

# Tensorflow Single Layer RNN
####################################################################
rnn_training = multiclass_rnn(trn_x = train_x_resized,
                              val_x = valid_x_resized,
                              tst_x = test_x_resized,
                              trn_y = train_y,
                              val_y = valid_y,
                              inv_class_wt = class_reweighting,
                              n_epochs = 100,
                              neurons = 3000,
                              early_stop = 12,
                              lrate = 0.002,
                              l1_reg = 0.015,
                              batch_size = 50,
                              save_dir = save_folder_path)

test_prediction = pd.concat([test_id, rnn_training], axis = 1)
test_prediction.columns = ['object_id','class_6','class_15','class_16','class_42','class_52','class_53',
                           'class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99']
test_prediction.to_csv(test_pred_path + "submission1.csv", index = False)





def pd_remaining_prob(df, excl_col = 'object_id'):
    return [(1 - i) for i in df.drop(excl_col, axis = 1).sum(axis = 0)]
    



















