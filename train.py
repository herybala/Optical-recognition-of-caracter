import os
from datetime import datetime as dt
import tensorflow as tf
import string
from utils.utils_fn import read_data, minibatch, variable, conv_block, dense_block, loss, parameter_update, accuracy_calc
from model.model import inference, train

bold = '\033[1m'
end = '\033[0m'

# Hyper-Parameters
path = "/home/supernova/Jobs/OCR/begining/ocr_model/"
data_folder = path + "dataset/test_records/"
max_char = 1
img_size = [128,128,1]
# digi, lc, sel uc, sel sign
checkpoint_restore = path + "checkpoints/checkpoint_digi_lc_sel_uc_sel_sign_1.ckpt"
checkpoint_save = path + "checkpoints/checkpoint_digi_lc_sel_uc_sel_sign_1.ckpt"
class_count = 56
keyword = 'digi_lc_sel_uc_sel_sign'
file_count = 1
train_data_count = 29400
batch_size = 1120
weights=[[3,3,1,16],
         [3,3,16,24],
         [3,3,24,42], 
         [3,3,42,56]]

train_filename =data_folder + 'dataset_' + keyword + '_'
test_filename = data_folder + 'dataset_' + keyword + '_'

dropout = [1, 1, 1, 1]
wd = 0.0

lr = 0.001
epochs = 10

num_of_threads=16
min_after_dequeue=10000
capacity = min_after_dequeue+(num_of_threads+1)*batch_size

# Train
train(data_folder, train_filename, test_filename,
	  train_data_count, file_count,
	  weights, dropout, wd,
	  img_size, max_char, class_count,
	  batch_size=batch_size, learning_rate=lr, epochs=epochs,
	  restore=False, var_lr=[None,None])