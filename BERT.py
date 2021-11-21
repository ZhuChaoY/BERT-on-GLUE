import re
import json
import time
import random
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm
from os import makedirs
from os.path import exists
from Process_GLUE import Process_GLUE
from Optimization import AdamWeightDecayOptimizer


class BERT():
    """A framework of BERT to training GLUE."""
    
    def __init__(self, args):
        """
        (1) Initialize BERT with args dict.
        (2) Named data dir and out dir.
        (3) Load entity, relation and triple.
        (4) Load common model structure.
        """
        
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))

        self.model_dir = 'Pretrained BERT/' + self.model + '/'
        self.out_dir = 'GLUE/{}/BERT-{}-{}/'.format(self.dataset, self.model,
                                                    self.len_d)
        if not exists(self.out_dir):
            makedirs(self.out_dir)
                            
        with open(self.model_dir + 'config.json') as file:
            config = json.load(file)
            self.dropout = config['hidden_dropout_prob']
            self.hidden = config['hidden_size']
            self.init_range = config['initializer_range']
            self.intermediate = config['intermediate_size']
            self.max_position = config['max_position_embeddings']
            self.n_head = config['num_attention_heads']
            self.n_layer = config['num_hidden_layers']
            self.type_vocab_size = config['type_vocab_size']
            self.vocab_size = config['vocab_size']
            
        self.initializer = tf.truncated_normal_initializer(self.init_range)

        print('\n\n' + '==' * 4 + ' < BERT-{} > && < {} > '.format(self.model,
             self.dataset) + '==' * 4)                 
        self.load_data()
        self.construct_model()
        
        
    def load_data(self):
        """Loading train and dev GLUE inputs."""
        
        inputs, self.n_label = Process_GLUE(self.dataset, self.len_d)
        for key in ['train', 'dev']:
            exec('self.' + key + " = inputs['" + key + "']")
            exec('self.n_' + key + ' = len(self.' + key + ')')
            print('    #{:5} : {}'.format(key, eval('self.n_' + key)))
    
    
    def construct_model(self):
        """Construct BERT model."""

        print('\n    #length of sequence : {}'.format(self.len_d))        
        print('    *Dropout_Rate       : {}'.format(self.dropout))
        print('    *Learning_Rate      : {}'.format(self.l_r))
        print('    *Batch_Size         : {}'.format(self.batch_size))
        print('    *Max_Epoch          : {}'.format(self.epoches))
        print('    *Earlystop Steps    : {}'.format(self.earlystop))
        
        tf.reset_default_graph()
        self.keep = tf.placeholder(tf.float32) 
        with tf.variable_scope('bert'):
            self.transformer_layer()
        with tf.variable_scope('loss'):
            self.finetune_layer()
            n_step = (self.n_train // self.batch_size + 1) * self.epoches
            self.train_op = AdamWeightDecayOptimizer(self.loss, self.l_r,
                                                     n_step, n_step // 10)
            
        
    def transformer_layer(self):    
        """BERT layer."""
        
        self.ids = tf.placeholder(tf.int32, [None, self.len_d]) #(B, L)
        ids = tf.reshape(self.ids, [-1]) #(B * L)
        self.segment = tf.placeholder(tf.int32, [None, self.len_d]) #(B, L)
        segment = tf.one_hot(tf.reshape(self.segment, [-1]), 2)
        #(1, 1, L, 1) * [(B, L) ==> (B, 1, 1, L)] ==> (B, 1, L, L)
        self.mask = tf.placeholder(tf.int32, [None, self.len_d]) #(B, L)
        att_mask = -10000.0 * (1.0 - tf.ones([1, 1, self.len_d, 1]) * \
                   tf.cast(tf.reshape(self.mask, [-1, 1, 1, self.len_d]),
                   tf.float32)) #(B, 1, 1, L)
        
        with tf.variable_scope('embeddings'): 
            #(vocab_size, H) ==> (B * L, H) ==> (B, L, H)
            word_table = tf.get_variable('word_embeddings', [self.vocab_size,
                         self.hidden], initializer = self.initializer)
            em_out = tf.reshape(tf.gather(word_table, ids), 
                                [-1, self.len_d, self.hidden]) 
            #(B*L,type_vocab_size)*(type_vocab_size,H)==>(B*L,H)==>(B,L,H)
            token_table = tf.get_variable('token_type_embeddings', 
                          [self.type_vocab_size, self.hidden],
                          initializer = self.initializer)
            em_out += tf.reshape(tf.matmul(segment, token_table), 
                                 [-1, self.len_d, self.hidden]) 
            #(B, L, H) + [(max_position, H) ==> (1, L, H)] ==> (B, L, H)
            position_table = tf.get_variable('position_embeddings',
                             [self.max_position, self.hidden],
                             initializer = self.initializer)
            em_out += tf.reshape(tf.slice(position_table, [0, 0],
                      [self.len_d, -1]), [1, self.len_d, self.hidden]) 
            em_out = self.dropout_layer(self.norm_layer(em_out))

        with tf.variable_scope('encoder'): #(B * L, H)
            prev_out = tf.reshape(em_out, [-1, self.hidden]) #(B * L, H)    
            for i in range(self.n_layer):
                with tf.variable_scope('layer_{}'.format(i)):
                    att_out = self.attention_layer(prev_out, att_mask)
                    prev_out = self.ffn_layer(att_out)
                    
        with tf.variable_scope('pooler'): #(B, H) 
            #(B * L, H) ==> (B, L, H) ==> (B, H) ==> (B, H)
            self.sequence_out = prev_out
            prev_out = tf.squeeze(tf.reshape(prev_out, [-1, self.len_d, 
                       self.hidden])[:, 0: 1, :], axis = 1)
            self.pooled_out = \
                self.dense_layer(prev_out, self.hidden, None, tf.tanh)
    
    
    def attention_layer(self, prev_out, att_mask):
        """Attention layer for bert layer"""
        
        with tf.variable_scope('attention'): #(B * L, H)
            with tf.variable_scope('self'): 
                #(B * L, H)=>(B * L, H)=>(B, L, head, 64)=>(B, head, L, 64)
                Q = self.dense_layer(prev_out, self.hidden, 'query')
                Q = tf.transpose(tf.reshape(Q, [-1, self.len_d,
                                 self.n_head, 64]), [0, 2, 1, 3])
                K = self.dense_layer(prev_out, self.hidden, 'key')
                K = tf.transpose(tf.reshape(K, [-1, self.len_d,
                                 self.n_head, 64]), [0, 2, 1, 3])
                V = self.dense_layer(prev_out, self.hidden, 'value')
                V = tf.transpose(tf.reshape(V, [-1, self.len_d,
                                 self.n_head, 64]), [0, 2, 1, 3])
                #(B, head, L, 64)*(B, head, 64, L)+(B, 1, L, L)==>(B,head,L,L)
                probs = self.dropout_layer(tf.nn.softmax(0.125 * tf.matmul(Q,
                        K, transpose_b = True) + att_mask))
                #(B, head, L, L) * (B, head, L, 64) ==> (B, head, L, 64) 
                # ==> (B, L, head, 64) ==> (B * L, H)   
                self_out = tf.reshape(tf.transpose(tf.matmul(probs, V), 
                           [0, 2, 1, 3]), [-1, self.hidden])
    
            with tf.variable_scope('output'): #(B * L, H)
                att_out = self.dense_layer(self_out, self.hidden)
                att_out = self.norm_layer(self.dropout_layer(att_out) + \
                                          prev_out)
        
        return att_out  
    
    
    def ffn_layer(self, att_out):
        """Feed Forward Network layer."""
        
        with tf.variable_scope('intermediate'): #(B * L, intermediate)
            mid_out = self.dense_layer(att_out, self.intermediate, None, gelu)
        with tf.variable_scope('output'): #(B * L, H)
            prev_out = self.dense_layer(mid_out, self.hidden)
            prev_out = self.norm_layer(self.dropout_layer(prev_out) + att_out)
        
        return prev_out
        
    
    def finetune_layer(self):     
        """Finetune layer for GLUE tasks."""
        
        w = tf.get_variable('output_weights', [self.n_label, self.hidden],
                            initializer = self.initializer)
        b = tf.get_variable('output_bias', [self.n_label],
                            initializer = tf.zeros_initializer())

        logits = tf.nn.bias_add(tf.matmul(tf.nn.dropout(self.pooled_out,
                                self.keep), w, transpose_b = True), b)
        if self.dataset == 'STS-B':
            logits = tf.squeeze(logits, [-1])
            self.label = tf.placeholder(tf.float32, [None])
            self.prediction = logits
            self.loss = tf.reduce_sum(tf.square(logits - self.label))
        else:
            self.label = tf.placeholder(tf.int32, [None]) 
            self.prediction = tf.argmax(tf.nn.softmax(logits, -1), -1)
            self.loss = tf.reduce_sum(-tf.reduce_sum(tf.one_hot(self.label,
                        self.n_label) * tf.nn.log_softmax(logits, -1), -1))  
        
                
    def norm_layer(self, _input):
        return tf.contrib.layers.layer_norm(inputs = _input,
               begin_norm_axis = -1, begin_params_axis = -1) 
    
    
    def dropout_layer(self, _input):
        return tf.nn.dropout(_input, self.keep)
    
    
    def dense_layer(self, _input, out_dim, name = None, activation = None):
        return tf.layers.dense(_input, out_dim, activation, name = name,
                               kernel_initializer = self.initializer)
    
    
    def _train(self, sess):
        """
        (1) Training process of BERT on GLUE.
        (2) Evaluate for dev dataset each epoch.
        """
        
        train_batches = self.get_batches('train')
        print('    EPOCH DEV-KPI  time   TIME (min)')
        result = {'args': self.args}
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(self.epoches):
            for ids, mask, segment, label in train_batches:
                feed_dict = {self.ids: ids, self.mask: mask,
                             self.segment: segment, self.label: label,
                             self.keep: 1.0 - self.dropout}
                _ = sess.run(self.train_op, feed_dict)
            kpi = self.cal_kpi(sess)    
            
            _t = time.time()
            print('    {:^5} {:^7.3f} {:^6.2f} {:^6.2f}'.format(ep + 1, kpi,
                  (_t - t1) / 60, (_t - t0) / 60), end = '')
            t1 = _t
            
            if ep == 0 or kpi > KPI[-1]:
                print(' *')
                if len(temp_kpi) > 0:
                    KPI.extend(temp_kpi)
                    temp_kpi = []
                KPI.append(kpi)
                tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
                result['dev-kpi'] = KPI
                result['best-epoch'] = len(KPI)
                with open(self.out_dir + 'result.json', 'w') as file: 
                    json.dump(result, file) 
            else:
                print('')
                if len(temp_kpi) == self.earlystop:
                    break
                else:
                    temp_kpi.append(kpi)
                
        if ep != self.epoches - 1:
            print('\n    Early stop at epoch of {} !'.format(len(KPI)))
                        
    
    def _predict(self, sess):
        """Prediction process."""
        
        dev_kpi = self.cal_kpi(sess)
        print('    DEV-KPI : {}'.format(dev_kpi))
            
        
    def cal_kpi(self, sess):
        """Calculate kpi for GLUE dataset."""
        
        dev_batches = self.get_batches('dev')

        Pre = None
        for ids, mask, segment, label in dev_batches:
            feed_dict = {self.ids: ids, self.mask: mask,
                         self.segment: segment, self.keep: 1.0}
            pre = sess.run(self.prediction, feed_dict)
            if Pre is None:
                Pre = pre
                Label = label
            else:
                Pre = np.hstack((Pre, pre))
                Label = np.hstack((Label, label))
                
        if self.dataset == 'CoLA':
            kpi = round(sm.matthews_corrcoef(Label, Pre), 4)
        elif self.dataset == 'STS-B':
            kpi = round(np.corrcoef(Label, Pre)[0][1], 4)
        else:
            acc = sum(Label == Pre) / Pre.shape[0]
            if self.dataset in ['MRPC', 'QQP']:
                f1 = sm.f1_score(Label, Pre)
                kpi = round((acc + f1) / 2, 4)
            else:
                kpi = round(acc, 4)
        
        return kpi
    
    
    def get_batches(self, key):    
        """Get input batches."""
        
        bs = self.batch_size
        data = eval('self.' + key + '.copy()')
        random.shuffle(data)                    
        n_batch = len(data) // bs
        idxes = [data[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if len(data) % bs != 0:
            idxes.append(data[n_batch * bs: ])
        batches = []
        for idx in idxes:
            ids = np.vstack([x[0] for x in idx])
            mask = np.vstack([x[1] for x in idx])
            segment = np.vstack([x[2] for x in idx])
            if key != 'test':
                label = np.array([x[3] for x in idx])
                batches.append((ids, mask, segment, label))
            else:
                batches.append((ids, mask, segment))
                
        return batches
    
    
    def initialize_variables(self, mode):
        """
        Initialize BERT structure trainable variables.
        
        Args:
            mode: 'train' or 'predict'
        """
        
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
        
        if mode == 'train':
            p = self.model_dir + 'model.ckpt'         
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs and 'bert' in v[0]}
        elif mode == 'predict':
            p = self.out_dir + 'model.ckpt'           
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
        tf.train.init_from_checkpoint(p, ivs)   
    
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        if self.do_train:
            print('\n>>  Training Process.')
            self.initialize_variables('train')        
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()   
                self._train(sess)
           
        if self.do_predict:
            print('\n>>  Predict Process.')
            self.initialize_variables('predict')   
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()  
                self._predict(sess)
    
    
def gelu(x):
    return x * 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) *
          (x + 0.044715 * x * x * x))))