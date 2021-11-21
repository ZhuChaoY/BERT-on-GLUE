import os
import argparse
import tensorflow as tf
from BERT import BERT


parser = argparse.ArgumentParser(description = 'Run GLUE')

parser.add_argument('--model', type = str, default = 'base',
                    help = 'model name') # 'base' or 'large'
# 'WNLI', 'RTE', 'MRPC', 'STS-B', 'CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI'
parser.add_argument('--dataset', type = str, default = 'WNLI',
                    help = 'dataset name') 
parser.add_argument('--len_d', type = int, default = 128,
                    help = 'length of sequence')
parser.add_argument('--dropout', type = float, default = 0.1, 
                    help = 'dropout rate')
parser.add_argument('--l_r', type = float, default = 2e-5, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 32, 
                    help = 'batch size')
parser.add_argument('--epoches', type = int, default = 10,
                    help = 'max training epoch')
parser.add_argument('--earlystop', type = int, default = 1,
                    help = 'training epoches')
parser.add_argument('--do_train', type = bool, default = True,
                    help = 'whether to train')
parser.add_argument('--do_predict', type = bool, default = True,
                    help = 'whether to predict')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True

model = BERT(args)
model.run(config)
    