# Import Packages
##############################################################################
import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns, os
import gc, pyodbc, random, tqdm, re, string, itertools, time
from string import punctuation
from sqlalchemy import create_engine, MetaData, Table, select
from os.path import getsize, basename
from __future__ import division
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from collections import Counter
from nltk.tokenize import RegexpTokenizer 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from tensorflow.python.client import timeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# User Input
##############################################################################
# Glove Embeddings
glove_txt_file = '********/glove.840B.300d.txt'
master_folder = '********/'
********_folders = []
********_folders = []

# Define Functions
###############################################################################
def seconds_to_time(sec):
    """convert seconds (integer or float) to time in 'hh:mm:ss' format"""
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

def sec_to_time_elapsed(end_tm, start_tm, return_time = False):
    """apply seconds_to_time function to start and end times"""
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    if return_time:
        return seconds_to_time(sec_elapsed)
    else:
        print('Execution Time: ' + seconds_to_time(sec_elapsed))

def unnest_list_of_lists(LOL):
    """unnest list of lists"""
    return list(itertools.chain.from_iterable(LOL))
    
def rem_multiple_substr(your_string, your_removals):
    """remove list of substrings ('your_removals') from a string ('your_string')"""
    your_new_string = your_string
    replace_dict = dict(zip(your_removals, ['' for yr in your_removals]))
    for removal, blank in replace_dict.items():
        your_new_string = your_new_string.replace(removal, blank)
    return your_new_string

def remove_http_https(some_string):
    """remove URLs starting with 'http' or 'https'"""
    return re.sub(r'http\S+', '', some_string, flags = re.MULTILINE)

def clean_tokenize(some_string):
    """split on punct / whitespace, make lower case, join back together"""
    pattern = re.compile(r'(\s+|[{}])'.format(re.escape(re.sub("'", "", punctuation))))
    clean_lower = ' '.join([part.lower() for part in pattern.split(remove_http_https(some_string)) if part.strip()])
    return clean_lower

def proc_line_delim_txt(folder_list,
                        prefix = master_folder,
                        postfix = '/page_links/',
                        encoding = 'utf-8'):
    """read and process text files in list of folder paths"""
    start_tm = time.time()
    paths = [prefix + i + postfix for i in folder_list]
    files = []
    for fpath in paths:
        files.append([fpath + x for x in os.listdir(fpath)])
    files = unnest_list_of_lists(files)
    txt = []
    for fpath in files:
        txt.append(open(fpath, encoding = "utf-8").read().splitlines())
    output = [clean_tokenize(t) for t in tqdm(unnest_list_of_lists(txt))]
    output_cust = []
    for t in output:
        tmp = re.sub("\'", "'", t)
        tmp = rem_multiple_substr(tmp, ["\\", "[ deleted ]", "https://", "https://", ".com", ".edu", "www.", "<hr/>", "/r/", "r/"])
        output_cust.append(tmp)
    end_tm = time.time()
    sec_to_time_elapsed(end_tm, start_tm)
    return output_cust

def remove_low_char_count(string_list, min_alpha_char = 10):
    """remove strings from list with less than 'min_alpha_char' alphabetic characters"""
    new_string_list = []
    rmv_string_count = []
    for s in tqdm(string_list):
        if len(re.sub("[^a-zA-Z]+", "", s)) >= min_alpha_char:
            new_string_list.append(s)
        else:
            rmv_string_count.append(1)
    n_rmv = str(int(sum(rmv_string_count)))
    print("removed " + n_rmv  + " observations with less than " + str(min_alpha_char) + " alphabetic characters")
    return new_string_list


def load_glove(glove_file_path, progress_print = 5000, encoding_type = 'utf8'):
    """load glove (Stanford NLP) file and return dictionary"""
    num_lines = sum(1 for line in open(glove_txt_file, encoding = encoding_type))
    embed_dict = dict()
    line_errors = []
    f = open(glove_file_path, encoding = encoding_type)
    for i, l in enumerate(f):
        l_split = l.split()
        try:
            embed_dict[l_split[0]] = np.asarray(l_split[1:], dtype = 'float32')
        except:
            line_errors.append(1)
        if ((i / progress_print) > 0) and (float(i / progress_print) == float(i // progress_print)):
            print(str(int(i / 1000)) + ' K of ' + str(int(num_lines / 1000)) + ' K lines completed')
        else:
            pass
    f.close()
    print('failed lines in file: ' + str(int(np.sum(line_errors))))
    return embed_dict

def glove_tokenize_proc(txt_doc, glove_dict, vocab_size = 100000, maxlen = 50):
    """ - read csv file with text in 'csv_txt_col' column"""
    """ - process into 300-dimension embeddings based on Stanford glove embeddings"""
    start_tm = time.time()
    tokenizer = Tokenizer(num_words = vocab_size, filters = '')
    tokenizer.fit_on_texts(txt_doc); print("tokenizer fit")
    sequences = tokenizer.texts_to_sequences(txt_doc)
    x_data = pad_sequences(sequences, maxlen = maxlen)
    embed_wt_matrix = np.zeros((vocab_size, 300))
    for x, i in tqdm(tokenizer.word_index.items()):
        if i > (vocab_size - 1):
            break
        else:
            embed_vec = glove_dict.get(x)
            if embed_vec is not None:
                embed_wt_matrix[i] = embed_vec
    end_tm = time.time()
    sec_to_time_elapsed(end_tm, start_tm)
    return x_data, embed_wt_matrix

# Import and Process Data
###############################################################################
********_txt = proc_line_delim_txt(folder_list = ********_folders)
********_txt = proc_line_delim_txt(folder_list = ********_folders)
********_txt_size_filt = remove_low_char_count(********_txt, min_alpha_char = ********)
********_txt_size_filt = remove_low_char_count(********_txt, min_alpha_char = ********)
del ********_txt; del ********_txt; gc.collect()
********_********_txt = ********_txt_size_filt + ********_txt_size_filt
y_dat = [1 for i in range(0, len(********_txt_size_filt))] + [0 for i in range(0, len(********_txt_size_filt))]

print("No. ******** ********s: " + str(len(********_txt_size_filt)))
print("No. ******** ********s: " + str(len(********_txt_size_filt)))
del ********_txt_size_filt; del ********_txt_size_filt; gc.collect()

# Load Glove Embeddings & Process Training Data
###############################################################################
glove_dict = load_glove(glove_txt_file)
x_dat, embed_wts = glove_tokenize_proc(txt_doc = ********, glove_dict = glove_dict, vocab_size = ********, maxlen = ********)
train_x, test_x, train_y, test_y = train_test_split(x_dat, y_dat, test_size = ********, random_state = ********)
del ********_txt; gc.collect()

# Model Fitting
###############################################################################
# Training on Two GPUs
K.clear_session()
train_start = time.time()
file_path = '********/keras_model.hdf5'
input_len = ********
embed_dim = ********
convd1_size = ********
conv_pool_size = ********
lstm_sizes = [********, ********, ********]
dropout_rates = [********, ********]
validation_perc = ********
num_epochs = ********
batch_size = ********
trn_pos_perc = sum(train_y) / len(train_y)
class_weight = {0: 1., 1: 1/trn_pos_perc}
vocab_size = ********
check_point = ModelCheckpoint(file_path,
                              monitor = 'val_loss',
                              verbose = 1,
                              save_best_only = True,
                              mode = 'min')
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 1)
model_glove = Sequential()
model_glove.add(Embedding(input_dim = vocab_size,
                          output_dim = embed_dim,
                          input_length = input_len,
                          weights = [embed_wts],
                          trainable = True))
model_glove.add(Dropout(dropout_rates[0]))
model_glove.add(Bidirectional(LSTM(lstm_sizes[0], return_sequences = True)))
model_glove.add(Dropout(dropout_rates[1]))
model_glove.add(Bidirectional(LSTM(lstm_sizes[1], return_sequences = True)))
model_glove.add(Bidirectional(LSTM(lstm_sizes[2])))
model_glove.add(Dense(1, activation = 'sigmoid'))
parallel_model = multi_gpu_model(model_glove, gpus = 2)
parallel_model.compile(loss='binary_crossentropy',
                       optimizer = optimizers.Adam(lr=********, beta_1=********, beta_2=********, epsilon=None, decay=0.0, amsgrad=False), 
                       metrics = ['accuracy'],
                       weighted_metrics = ['accuracy'])
parallel_model.fit(train_x,
                   np.array(train_y),
                   validation_split = validation_perc,
                   epochs = num_epochs,
                   batch_size = batch_size,
                   class_weight = class_weight,
                   callbacks = [check_point, early_stop])

train_end = time.time()
sec_to_time_elapsed(train_end, train_start)
test_pred = parallel_model.predict(test_x)
test_df = pd.DataFrame({'actual': test_y,
                        'pred_prob': [p[0] for p in test_pred],
                        'grouping': ['All' for p in test_pred]})
    