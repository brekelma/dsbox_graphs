import os
import sys
import typing
import networkx
import numpy as np
import pdb
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras
import pandas as pd
import copy 
import importlib
import collections

import keras.objectives
import keras.backend as K
#from sklearn import preprocessing
import tempfile
import scipy.sparse
from scipy.sparse import csr_matrix
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import keras.models

from common_primitives import utils
import d3m.container as container
import d3m.metadata.base as mbase
from d3m.base import utils as base_utils
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import List as d3m_List
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import _config as cfg_
#import config as cfg_


Input = typing.Union[container.List, container.DataFrame]
Output = container.DataFrame
                            
     
import logging
_logger = logging.getLogger(__name__)

# all primitives must be pickle-able, and this should do the trick for Keras models

def loss_fun(inputs, function = None, first = None):
        if isinstance(function, str):
                import importlib
                mod = importlib.import_module('keras.objectives')
                function = getattr(mod, function)
        #try:
        print("*"*500)
        print("LOSS FUNCTION ", function)
        print("inputs ", inputs)
        print("*"*500)

        return function(inputs[0], inputs[-1]) #if function is not None else inputs
        #except:
        #        inputs[0] = tf.gather(inputs[0], np.arange(first))
        #        return function(inputs[0], inputs[-1]) #if function is not None else inputs



def dummy_concat(inputs, total = None, keep = None):
        tensor = inputs[0]
        shape_ref = inputs[-1]
        dummy = tf.zeros_like(shape_ref)[:(tf.shape(shape_ref)[0] - tf.shape(tensor)[0])]

        try:
                return tf.concat([tensor, dummy], axis = 0, name = 'dummy_concat')
        except:
                return tf.concat([tf.expand_dims(tensor,-1), dummy], axis = 0, name = 'dummy_concat')

def assign_scattered(inputs):
        slice_loss = inputs[0]
        shape_ref = inputs[1]
        inds = tf.expand_dims(tf.cast(inputs[-1], tf.int32), -1)
        full_loss = tf.scatter_nd(inds, slice_loss, shape = [tf.shape(shape_ref)[0]])
        return full_loss #tf.reshape(full_loss, (-1,))

def identity(x_true, x_pred):
        return x_pred

def dot(x, y, sparse=False):
        """Wrapper for tf.matmul (sparse vs dense)."""
        if sparse:
                try:
                        res = tf.sparse_tensor_dense_matmul(x, y)
                except:
                        x = tf.contrib.layers.dense_to_sparse(x)
                        res = tf.sparse_tensor_dense_matmul(x, y)
        else:
                res = tf.matmul(x, y)
        return res

def sparse_exp_ax(adj, x, exponent = 1):
        res = x
        if exponent == 0:
                return res
       
        for k in range(exponent):
                res = dot(adj, res, sparse = True)
        return res

def get_columns_of_type(df, semantic_types): 
        columns = df.metadata.list_columns_with_semantic_types(semantic_types)

        def can_use_column(column_index: int) -> bool:
                return column_index in columns

        # hyperparams['use_columns'], hyperparams['exclude_columns']
        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(df.metadata, [], [], can_use_column) # metadata, include, exclude_columns, idx_function

        if not columns_to_use:
                raise ValueError("Input data has no columns matching semantic types: {semantic_types}".format(
                        semantic_types=semantic_types,
                ))

        #if columns_not_to_use: #and hyperparams['use_columns']
                #cls.logger.warning("Node attributes skipping columns: %(columns)s", {
                #        'columns': columns_not_to_use,
                #})
        try:
                ret = df.select_columns(columns_to_use)
        except:
                ret = df.select_columns([i for i in range(len(df.columns)) if 'attr' in df.columns[i].lower()])
            #try:
            #except:
            #    ret = df.select_columns(columns_to_use)[:-2]
        return ret

class Slice(keras.layers.Layer):
        def __init__(self, **kwargs):
                super(Slice, self).__init__(**kwargs)

        def build(self, input_shape):
                s = list(input_shape[0])
                s[0] = None
                self.shape = s
                self.built = True

        def call(self, inputs):
                tensor = inputs[0]
                inds = inputs[-1]
                sliced = tf.gather(tensor, tf.squeeze(inds), axis = 0)
                sliced.set_shape(self.shape)
                return sliced

        def compute_output_shape(self, input_shape):
                #return input_shape[0]
                return self.shape


class GCN_Layer(keras.layers.Layer):
        def __init__(self, k = 1, adj = None, pre_w = None, **kwargs):
                self.k = k
                self.adj = adj
                self.pre_w = None
                super(GCN_Layer, self).__init__(**kwargs)

        def build(self,input_shape):
                self.built = True
                
        def call(self, inputs):
                if isinstance(inputs,list):
                        self.adj = inputs[-1]
                        x = inputs[0]
                #else:
                #        x = inputs

                ret=  sparse_exp_ax(self.adj, x, exponent = self.k)
                return ret
                
        def compute_output_shape(self, input_shape):
                shape = input_shape if not isinstance(input_shape, list) else input_shape[0]
                if isinstance(input_shape, list):
                        shape = list(shape)
                        shape[0] = input_shape[-1][0]
                        shape = tuple(shape)
                return shape

#def gcn_layer(inputs,  pre_w = None, k = 1, adj = None):
#        print("INPUTS ", inputs[0], inputs[-1])
#        # keras wrapper for sparse mult
#        # exponent for mixhop  
#       return sparse_exp_ax(adj, x, exponent = k)      


class GCN_Network(object):
        def __init__(self, nodes = None, units = [100, 100], mix_hops = 2, modes = 1, lr = 0.0005, extra_fc = 100, 
                        outputs_tensor = None, inds_tensor = None, sparse = True, epochs = 100, logger = None,
                        labeled = None, input_columns = None, label_unique = None):
                self._num_training_nodes = nodes
                self.outputs_tensor = outputs_tensor
                self.inds_tensor =inds_tensor
                self.sparse = sparse

                self._num_labeled_nodes = labeled 
                self._input_columns = input_columns
                self._label_unique = label_unique
                self._input_columns = input_columns
                
                self.logger = logger

                #self.adj_inds = tf.placeholder(tf.int64)
                #self.adj_vals = tf.placeholder(tf.float32)
                #self.adj_input = tf.SparseTensor(self.adj_inds, self.adj_vals, dense_shape = (self._num_training_nodes, self._num_training_nodes))


                self.keras_fit = False
                self._task = 'classification'
                self._act = 'relu'
                self._epochs = epochs #self.hyperparams['epochs'] #200 if self._num_training_nodes < 10000 else 50
                self._units = units #[self.hyperparams['layer_size']]*self.hyperparams['layers']
                self._mix_hops = mix_hops #self.hyperparams['adjacency_order']
                self._modes = modes
                self._lr = lr
                #self._optimizer = keras.optimizers.Adam(lr)
                self._extra_fc = extra_fc


                self.y_true = self.outputs_tensor #keras.layers.Input(tensor = self.outputs_tensor, name = 'y_true', dtype = 'float32')
                self.inds = self.inds_tensor #keras.layers.Input(tensor = self.inds_tensor, dtype='int32', name = 'training_inds')


                if self.sparse:
                        self.adj_inds = tf.placeholder(tf.int64)
                        self.adj_vals = tf.placeholder(tf.float32)
                        self.adj_input = tf.SparseTensor(self.adj_inds, self.adj_vals, dense_shape = (self._num_training_nodes, self._num_training_nodes))
                else:
                        self.adj_input = tf.sparse_placeholder(tf.float32, shape = (self._num_training_nodes, self._num_training_nodes))
                
                #print("ADJ INPUT ", self.adj_input)
                self.feature_input = tf.placeholder(tf.float32, shape = (self._num_training_nodes, self._input_columns)) #tf.sparse_placeholder(tf.float32, shape = (self._num_training_nodes, self._num_training_nodes))
                #print("FEATURE INPUT ", self.feature_input)
                # using self.y_true and self.inds right now...
                #self.y = tf.placeholder(tf.float32, [None, self.training_outputs.shape[-1]], name='y')
                #self.ph_indices = tf.placeholder(tf.int64, [None])
                

                #feature_input = keras.layers.Input(shape = self._input.shape[1:], name = 'features')#sparse =True) #tensor =  if tensor else tf.convert_to_tensor(to_return))
                #adj_normalize = tf.divide(tf.sum(self.adj_input, axis = -1))
                self.embedding = self._make_gcn(self.adj_input, self.feature_input,
                        h_dims = self._units,
                        mix_hops = self._mix_hops, 
                        modes = self._modes)

                if self._extra_fc is not None:
                        self.embedding = tf.layers.Dense(self._extra_fc, activation = self._act)(self.embedding)

                label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
                print("LABEL ACT", label_act)
                self.y_pred = tf.layers.Dense(self._label_unique, activation = label_act, name = 'y_pred')(self.embedding)
        
                outputs = []
                loss_functions = []
                loss_weights = []
        
                #outputs.append(y_pred)

                if label_act == 'softmax':  # if self._task == 'node_clf': 
                        #loss_functions.append(keras.objectives.categorical_crossentropy)
                        loss_function = keras.objectives.categorical_crossentropy
                else: #f label_act == 'sigmoid':
                        #loss_functions.append(keras.objectives.binary_crossentropy)
                        loss_function = keras.objectives.binary_crossentropy
                #    loss_functions.append(keras.objectives.mean_squared_error)#mse                                            


                self.embedding_slice = self.semi_supervised_slice([self.embedding, self.inds])
                     #keras.layers.Lambda(self.semi_supervised_slice)([self.embedding, self.inds])
                # Note: Y-true is an input tensor
                self.y_pred_slice = self.semi_supervised_slice([self.y_pred, self.inds])
                     #keras.layers.Lambda(self.semi_supervised_slice)([self.y_pred, self.inds])#, arguments = {'inds': self.training_inds})(y_pred)
                # doesn't acutally use total / keep
                self.y_true_slice = self.semi_supervised_slice([self.y_true, self.inds])
                     #keras.layers.Lambda(self.semi_supervised_slice)([self.y_true, self.inds])
                
                print("*"*500)
                print(loss_function)
                print(self.y_true_slice, self.y_pred_slice)
                self.slice_loss = loss_fun([self.y_true_slice, self.y_pred_slice], function = loss_function, first = self._num_labeled_nodes)
                    #keras.layers.Lambda(loss_fun, arguments = {'function': loss_function, 'first': self._num_labeled_nodes})([self.y_true_slice, self.y_pred_slice])
                
                #full_loss = assign_scattered([self.slice_loss, self.y_pred, self.inds])
                #keras.layers.Lambda(assign_scattered)([self.slice_loss, self.y_pred, self.inds])


                #tf_loss = self.slice_loss #full_loss

                self.learn_rate = tf.placeholder(tf.float32, [], 'learn_rate')
                self.optimizer = tf.train.MomentumOptimizer(self.learn_rate, 0.7, use_nesterov=True)
                self.train_op = slim.learning.create_train_op(
                        self.slice_loss, self.optimizer, gradient_multipliers=[])

                
                #self.saver = tf.train.Saver() 
                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())

                        
        def fit(self, adj, features):
                # STOPPING AFTER CERTAIN AMOUNT OF TIME?

                tic = time.time()
                for i in range(self._epochs):
                        # TO DO : anneal learning rate
                        try:
                                if i > int(self._epochs/2):
                                        lr = self._lr*(1- (i - int(self._epochs/2))*1.0/(self._epochs - int(self._epochs/2)))
                                else:
                                        lr = self._lr
                        except:
                                lr = self._lr
                        preds, loss_value = self.step(adj, features, lr)
                        self.logger.info(str("Epoch "+str(i)+"Loss "+str(np.mean(loss_value))))
                        print("Epoch ", i, " Loss ", np.mean(loss_value))
                        if time.time()-tic > 3000:
                                break


        def semi_supervised_slice(self, inputs, first = None):
                        if isinstance(inputs, list):
                                tensor = inputs[0]
                                inds = inputs[-1]
                                inds = tf.squeeze(inds)
                        #else:
                        #        tensor = inputs
                        #        inds = np.arange(first, dtype = np.float32)
                        #try:
                        sliced = tf.gather(tensor, inds, axis = 0)
                        #except:
                        #        sliced = tf.gather(tensor, tf.cast(inds, tf.int32), axis = 0)
                        #sliced.set_shape([None, sliced.get_shape()[-1]])
                        return tf.cast(sliced, tf.float32)
                        #return tf.cast(tf.reshape(sliced, [-1, tf.shape(tensor)[-1]]), tf.float32)
                        
        def step(self, adj, features, lr=None, columns=None):
            #i = LAST_STEP['step']
            #LAST_STEP['step'] += 1
            #feed_dict[is_training] = True
            #feed_dict[ph_indices] = train_indices



            if not isinstance(adj, list):
                    feed_dict = {self.adj_input: adj,       
                                 self.feature_input: features}
            else:
                    feed_dict = {self.adj_inds: adj[0],
                                 self.adj_vals: adj[-1],
                                 self.feature_input: features}


                #y: self.training_outputs
                #}
            if lr is not None:
              feed_dict[self.learn_rate] = lr
            #print("FEED ", feed_dict)            
                # Train step
            #train_preds, loss_value, _ = sess.run((sliced_output, label_loss, train_op), feed_dict)


            _, train_preds, loss_value, pred, y_true, y_true_slice = self.sess.run((self.train_op, self.y_pred_slice, self.slice_loss, self.y_pred, self.y_true, self.y_true_slice), feed_dict)

            #print("train preds ", train_preds)
            #print('pred ', pred)
            #print("true ", y_true)

            #import IPython; IPython.embed()


            if np.isnan(loss_value).any():
                    print('NaN value reached. Debug please.')
                    import IPython; IPython.embed()

            return train_preds, loss_value

        def _make_gcn(self, adj, features, h_dims = [100, 100], mix_hops = 5, modes = 1):
                a = adj
                x = features #self._input
                # where should number of modes be handled?
                for h_i in range(len(h_dims)):
                        act_k = []

                        for mode in range(modes):
                                for k in range(mix_hops+1):
                                        #pre_w = keras.layers.Lambda(gcn_layer, arguments = {'k': k})([x, self._adj])
                                        #pre_w = tf.sparse.to_dense(sparse_exp_ax(a, x, exponent = k))
                                        #pre_w = tf.sparse_tensor_to_dense(sparse_exp_ax(a, x, exponent = k))
                                        pre_w = sparse_exp_ax(a, x, exponent = k)
                 
                                        #print("PRE W ")
                                        #print(pre_w)
                                        #pre_w = GCN_Layer(k = k)([x,a])#, adj = a)(x)
                                        #import IPython; IPython.embed()
                                        #keras.layers.Lambda(function, arguments = {'k': k})(x)
                                        act = tf.layers.Dense(h_dims[h_i], activation = self._act, name='w'+str(k)+'_'+str(mode)+'_'+str(h_i))(pre_w)
                                        act_k.append(act)
                                        
                        x = tf.concat(act_k, name = 'mix_'+str(mix_hops)+'hops_'+str(h_i), axis = -1) #keras.layers.Concatenate(axis = -1, name = 'mix_'+str(mix_hops)+'hops_'+str(h_i))(act_k)
            
                # embedding tensor (concatenation of k)
                return x

        def pred(self, adj_input, feature_input, embedding = False, _slice = False):
                if isinstance(adj_input, list):
                        fd = {self.adj_inds: adj_input[0],
                                     self.adj_vals: adj_input[-1],
                                     self.feature_input: feature_input}
  
                   
                else:
                        fd = {self.adj_input: adj_input,       
                              self.feature_input: feature_input}#,
                
                if not embedding:
                        if not _slice:
                                output = self.sess.run([self.y_pred], feed_dict = fd)
                        else:
                                output = self.sess.run([self.y_pred_slice], feed_dict = fd)
                else:
                        if not _slice:
                                output = self.sess.run([self.embedding], feed_dict = fd)
                        else:
                                output = self.sess.run([self.embedding_slice], feed_dict = fd)
                return output if not isinstance(output,list) else output[0]


class GCN_Params(params.Params):

        ''' 
        Attributes necessary to resume training or run on test data (if loaded from pickle)

        Code specifications of parameters: 
                https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/params.py
        '''

        fitted: typing.Union[bool, None] # fitted required, set once primitive is trained
        #model: keras.models.Model #typing.Union[keras.models.Model, None]d
        network: GCN_Network
        node_encode: typing.Union[LabelEncoder, None]
        label_encode: typing.Union[LabelEncoder,None]
        #pred_model: keras.models.Model
        #embed_model: keras.models.Model
        #weights: typing.Union[typing.Any, None]
        #pred_weights: typing.Union[typing.Any, None]
        #embed_weights: typing.Union[typing.Any, None]
        #adj: typing.Union[tf.Tensor, tf.SparseTensor, tf.Variable, keras.layers.Input, np.ndarray, csr_matrix, None]

class GCN_Hyperparams(hyperparams.Hyperparams):

        ''' 
        Code specifications of hyperparameters: 
                https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/hyperparams.py
        '''

        dimension = UniformInt(
                lower = 10,
                upper = 200,
                default = 100,
                description = 'dimension of latent embedding',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )
        adjacency_order = UniformInt(
                lower = 1,
                upper = 5,
                default = 3,
                #q = 5e-8,
                description = 'Power of adjacency matrix to consider.  1 recovers Vanilla GCN.  MixHop (Abu El-Haija et al 2019) performs convolutions on A^k for 0 <= k <= order and concatenates them into a representation, allowing the model to consider k-step connections.',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )

        # hidden_layers
        
        # epochs
        epochs = UniformInt(
                lower = 50,
                upper = 10000,
                default = 200,
                #q = 5e-8,                                                                                                                                                                 
                description = 'number of epochs / gradient steps to train (entire graph each batch)',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )

        return_embedding = UniformBool(
                default = True,
                description='return embedding alongside classification prediction (can be used as classifier otherwise)',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
        )
        
        line_graph = UniformBool(
                default = False,
                description='treat edges as nodes, construct adjacency matrix based on shared edges.  relevant for edge based classification, e.g. link prediction',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
        )
        layers = UniformInt(
                lower = 1,
                upper = 10,
                default = 2,
                description = 'number of layers',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )
        layer_size = UniformInt(
                lower = 10,
                upper = 200,
                default = 50,
                description = 'units per layer',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )
        lr = Uniform(
                lower = .0001,
                upper = 1,
                default = .01,
                description = 'learning rate',
                semantic_types=["http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )
        




def make_keras_pickleable():
        def __getstate__(self):
                #setattr(gcn_layer, '__deepcopy__', lambda self, _: self)
                #setattr(GCN, '__deepcopy__', lambda self, _: self)

        
                model_str = ""
                model_weights = ""
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                        #self.save(fd.name)#, overwrite=True)
                        keras.models.save_model(self, fd.name, overwrite=True)
                        model_str = fd.read()
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
                        self.save_weights(fd.name)
                        model_weights = fd.read()
                d = {'model_str': model_str, 'model_weights': model_weights}
                return d

        def __setstate__(self, state):
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                        fd.write(state['model_str'])
                        fd.flush()
                        model = keras.models.load_model(fd.name, custom_objects = 
                                {'tf': tf, 'GCN_Layer': GCN_Layer, 'identity': identity,
                                'assign_scattered': assign_scattered})
                        #model.load_weights(
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                        fd.write(state['model_weights'])
                        fd.flush()
                        model.load_weights(fd.name)
                self.__dict__ = model.__dict__
                
        
        #cls = Sequential
        #cls.__getstate__ = __getstate__
        #cls.__setstate__ = __setstate__

        cls = keras.models.Model
        cls.__getstate__ = __getstate__
        cls.__setstate__ = __setstate__




class GCN(SupervisedLearnerPrimitiveBase[Input, Output, GCN_Params, GCN_Hyperparams]):
        """
        See base classes here : 
                https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/primitive_interfaces

        """

        metadata = PrimitiveMetadata({
                "schema": "v0",
                "id": "48572851-b86b-4fda-961d-f3f466adb58e",
                "version": "1.0.0",
                "name": "GCN",
                "description": "Graph convolutional neural networks (GCN) as in Kipf & Welling 2016, generalized to k-hop edge links via Abu-el-Haija et al 2019: https://arxiv.org/abs/1905.00067 (GCN recovered for k = 1).  In particular, learns weight transformation of feature matrix X for various powers of adjacency matrix, i.e. nonlinearity(A^k X W), and concatenates into an embedding layer.  Feature input X may be of the form: identity matrix (node_id) w/ node features appended as columns.  Specify order using 'adjacency_order' hyperparam.  Expects list of [learning_df, nodes_df, edges_df] as input (e.g. by running common_primitives.normalize_graphs + data_tranformation.graph_to_edge_list.DSBOX)",
                #"python_path": "d3m.primitives.feature_construction.gcn_mixhop.DSBOX",
                "python_path": "d3m.primitives.feature_construction.graph_transformer.GCN",
                "original_python_path": "gcn_mix.GCN",
                "source": {
                        "name": "ISI",
                        "contact": "mailto:brekelma@usc.edu",
                        "uris": [ "https://github.com/brekelma/dsbox_graphs" ]
                },
                "installation": [ cfg_.INSTALLATION ],

                # See possible types here :https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json
                "algorithm_types": ["CONVOLUTIONAL_NEURAL_NETWORK"],
                "primitive_family": "FEATURE_CONSTRUCTION",
                "hyperparams_to_tune": ["dimension", "adjacency_order"]
        })
        
        def __init__(self, *, hyperparams : GCN_Hyperparams) -> None:
                super(GCN, self).__init__(hyperparams = hyperparams)

        
        def set_training_data(self, *, inputs : Input, outputs : Output) -> None:
                #self._adj = None
                #self._input = None

                learning_df, nodes_df, edges_df = self._parse_inputs(inputs)
                
                self.node_encode = None
                self._adj, self._input, targets = self._get_training_data(learning_df, nodes_df, edges_df)
                
                #if not self.hyperparams['line_graph']:        
                #        print(edges_df[targets[0]])
                
                #targets_np = np.zeros(nodes_df.shape[0])
                #targets_np[learning_df['d3mIndex'].astype(np.int32).values] = learning_df[targets.columns].astype(np.int32).values
                #nodes_df.loc[result_df.index.isin(learning_df['d3mIndex'].values)]
                #print("Full targets ", targets_np)
                #targets = targets_np
        
                #self.training_outputs = to_categorical(self.label_encode.fit_transform(outputs), num_classes = np.unique(outputs.values).shape[0])
                #else:
                #        raise NotImplementedError()
                print("Learning df shape ")
                print(learning_df.shape)
                print("targets ")
                print(targets.shape)
                
                self._set_training_values(learning_df, targets)
                
                self.fitted = False

        def _get_training_data(self, learning_df, nodes_df, edges_df, node_subset = None):
                if self.hyperparams['line_graph']:
                        self._num_training_nodes = edges_df.values.shape[0]
                        _adj = self._make_line_adj(edges_df)
                        _input = self._make_line_inputs(edges_df)
                else:

                        #nodes_df = nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]
                        #node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]

                        # try:
                        #         self._num_training_nodes = node_subset.values.shape[0]
                        # except:
                        #         self._num_training_nodes = nodes_df.values.shape[0]
                        #self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0], tensor = True)#, node_subset = node_subset.values.astype(np.int32))
                        #self._input = self._make_input_features(nodes_df, tensor = True)#.loc[learning_df['d3mIndex'].astype(np.int32)])#.index])
                        self.sparse = True

                        # should be gone
                        #if self.node_encode is None:
                            #self.node_encode = LabelEncoder()
                            #num_nodes = nodes_df.shape[0], 
                        _adj = self._make_adjacency(edges_df, sparse = self.sparse) #tensor = True)#, node_subset = node_subset.values.astype(np.int32))
                        
                        # NODE ENCODE
                        print(nodes_df)
                        print("VALUES")
                        print(nodes_df['nodeID'].values)
                        print("AS INT")
                        print(nodes_df['nodeID'].astype(np.int32).values)
                        try:
                                nodes_df['nodeID']  = self.node_encode.transform(nodes_df['nodeID'].astype(np.int32).values)
                        except:
                                pass
                        
                        _input = self._make_input_features(nodes_df, num_nodes = self._num_training_nodes)#, tensor = True)#.loc[learning_df['d3mIndex'].astype(np.int32)])#.index])

                # dealing with outputs
                #if self._task in ['clf', 'class', 'classification', 'node_clf']:

                target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                  
                use_outputs = False
                if use_outputs:
                        targets =  get_columns_of_type(outputs, target_types)
                else:
                        targets =  get_columns_of_type(learning_df, target_types)

                return _adj, _input, targets

        def _set_training_values(self, learning_df, targets):
                self.training_inds = learning_df['d3mIndex'].astype(np.int32).values
                print()
                print("TRAINING INDS")
                print(self.training_inds)
                self.training_inds = self.node_encode.transform(learning_df['d3mIndex'].astype(np.int32).values)
                print(self.training_inds)

                self._label_unique = np.unique(targets.values).shape[0]
                #self._label_unique = np.unique(targets).shape[0]
                try:
                    self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                    self.label_encode = None
                except:
                    self.label_encode = LabelEncoder()
                    self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                
                #self.training_outputs = to_categorical(self.label_encode.fit_transform(targets), num_classes = np.unique(targets).shape[0])
                
                self._num_labeled_nodes = self.training_outputs.shape[0]
                
                # OPTION TO NOT FILL IN TO FULL RANK
                print("*"*50)
                print(self._num_training_nodes)
                print('training outputs ', self.training_outputs.shape)
                training_outputs = np.zeros(shape = (self._num_training_nodes, self.training_outputs.shape[-1]))
                #try:
                for i in range(self.training_inds.shape[0]):
                       print("SETTING ", self.training_inds[i])
                       training_outputs[self.training_inds[i],:]= self.training_outputs[i, :]
                self.training_outputs = training_outputs
                print("training outputs" )
                print(training_outputs)
                self.outputs_tensor = tf.constant(self.training_outputs)
                self.inds_tensor = tf.constant(np.squeeze(self.training_inds), dtype = tf.int32)

                self.y_true = self.outputs_tensor # keras.layers.Input(tensor = self.outputs_tensor, name = 'y_true', dtype = 'float32')
                self.inds = self.inds_tensor #keras.layers.Input(tensor = self.inds_tensor, dtype='int32', name = 'training_inds')

        def _get_source_dest(self, edges_df, source_types = None, dest_types = None):
                if source_types is None:
                        source_types = ('https://metadata.datadrivendiscovery.org/types/EdgeSource',
                                        'https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource',
                                        'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource',
                                        'https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource',
                                        'https://metadata.datadrivendiscovery.org/types/MultiEdgeSource')

                #sources = edges_df['source']
                sources = get_columns_of_type(edges_df, source_types)
                
                if dest_types is None:
                        dest_types = ('https://metadata.datadrivendiscovery.org/types/EdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget')
                dests = get_columns_of_type(edges_df, dest_types)
                
                return sources.astype(np.int32), dests.astype(np.int32)


        def _normalize_adj(self, adj):
                try:
                        diag = np.diag(adj.sum(axis = -1))
                        diag = csr_matrix(diag**(-.5))
                except:
                        diag = np.diag(np.sum(adj, axis = -1))
                
                normalized = diag.dot(adj + csr_matrix(np.eye(adj.shape[-1]))).dot(diag)
                #np.dot(np.dot(diag**(-.5), adj + np.eye(adj.shape[-1])), diag**(-.5))
                return normalized


        def _make_line_adj(self, edges_df, node_subset = None, tensor = False):
                sources, dests = self._get_source_dest(edges_df)
                
                num_nodes = edges_df.shape[0]

                # TO DO: change edge detection logic to reflect directed / undirected edge source/target
                #   multigraph = different adjacency matrix for each edge type?  
                # to do: various different edge weights?
                
                # edges indexed by rows / index of edges_df
                edges = [[i,j] for i in range(sources.values.shape[0]) for j in range(dests.values.shape[0]) if dests.values[j,0] == sources.values[i,0]]
                weights = [1.0 for i in range(len(edges))]
                

                # label encoding of nodes not necessary (only looking for shared nodes)
                # label encoding of edges not necessary if edges have unique ID ordered 0:num_edges
                #self.node_enc = LabelEncoder()
                #node_subset = node_subset if node_subset is not None else edges_df['d3mIndex'].values
                #self.node_enc.fit(node_subset) # edges indices

                if tensor:
                        adj = tf.SparseTensor(edges, weights, dense_shape = (num_nodes, num_nodes))
                else:
                        edges = ([edges[i][0] for i in range(len(edges))], [edges[i][1] for i in range(len(edges))])
                        adj = csr_matrix((weights, edges), shape = (num_nodes, num_nodes), dtype = np.float32)

                
        def _make_adjacency(self, edges_df, num_nodes = None, tensor = False, sparse = False, #True, 
                            node_subset = None):
                
                sources, dests = self._get_source_dest(edges_df)

                #attr_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                #                          'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')
                #attrs = get_columns_of_type(edges_df, attr_types)
                
                #attrs = self.node_enc.transform(attrs.value)

                #self.node_enc = LabelEncoder()
                #id_col = [i for i in nodes_df.columns if 'node' in i and 'id' in i.lower()][0]
                to_fit = node_subset if node_subset is not None else np.concatenate([sources.astype(np.int32).values,dests.astype(np.int32).values], axis = -1).ravel()
                for i in range(to_fit.shape[-1]):
                        s = list(sources.columns)[-1]
                        d = list(dests.columns)[-1]
                        sources.append({s: i}, ignore_index = True)
                        dests.append({d: i}, ignore_index = True)

                if self.node_encode is None:
                    self.node_encode = LabelEncoder()
                    self.node_encode.fit(to_fit)
                    try:
                            self._num_training_nodes = self.node_encode.classes_.shape[0]
                    except:
                            self._num_training_nodes = np.unique(to_fit).shape[0]
                    num_nodes = self._num_training_nodes
                    
                    print("NODES PRE-ENCODE ", to_fit)
                    print('NODES ENCODED ', self.node_encode.transform(to_fit))
                    #print('encoding clases ', self.node_encode.classes_.shape[0])
                else:
                    pass #self.node_encode.transform(to_fit) #nodes_df[id_col].values)


                if node_subset is not None:
                        node_subset = node_subset.values if isinstance(node_subset,pd.DataFrame) else node_subset 
                        num_nodes = node_subset.shape[0]

                        inds = [i for i in sources.index if sources.loc[i, sources.columns[0]] in node_subset and dests.loc[i, dests.columns[0]] in node_subset]

                        
                        sources = sources.loc[inds] 
                        dests = dests.loc[inds]
                        #attrs = attrs.loc[inds]
                
                # ADD SELF CONNECTIONS IN ADJACENCY MATRIX
                sources[sources.columns[0]] = self.node_encode.transform(sources.astype(np.int32).values)
                dests[dests.columns[0]] = self.node_encode.transform(dests.astype(np.int32).values)


                print("*"*50)
                print("sources post transform ", sources)
                print("dests ", dests)
                print("*"*50)
                # accomodate weighted graphs ??
                if tensor:
                        adj = tf.SparseTensor([[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])], [1.0 for i in range(sources.values.shape[0])], dense_shape = (num_nodes, num_nodes))
                        #adj = tf.SparseTensorValue([[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])], [1.0 for i in range(sources.values.shape[0])], dense_shape = (num_nodes, num_nodes))
                elif sparse:
                        sparse_inds = [[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])]
                        sparse_values = [1.0/np.sum(sources.values == sources.values[i]) for i in range(sources.values.shape[0])]
                else:
                        adj = csr_matrix(([1.0 for i in range(sources.values.shape[0])], ([sources.values[i, 0] for i in range(sources.values.shape[0])], [dests.values[i,0] for i in range(sources.values.shape[0])])), shape = (num_nodes, num_nodes), dtype = np.float32)
                #tf.sparse.placeholder(dtype,shape=None,name=None)

                
                return adj if not sparse else [sparse_inds, sparse_values]
                # PREVIOUS RETURN
                #self._adj = keras.layers.Input(tensor = adj if tensor else tf.convert_to_tensor(adj), sparse = True)

                #if self._adj is None:                
                #else:
                        #self._adj = #tf.assign(self._adj, adj if tensor else tf.convert_to_tensor(adj))
                        

                #raise NotImplementedError
                #return keras.layers.Input()

        def _make_line_inputs(self, nodes_df, tensor = False):
                # feature attributes

                # ID for evaluating adjacency matrix
                if tensor:
                        node_id = tf.cast(tf.eye(nodes_df.shape[0]), dtype = tf.float32)
                        #node_id = tf.sparse.eye(nodes_df.shape[0])
                else:
                        #node_id = scipy.sparse.identity(nodes_df.shape[0], dtype = np.float32) #
                        node_id = np.eye(nodes_df.shape[0])
                
                # additional features?
                # preprocess features, e.g. if non-numeric / text?
                if False:# len(nodes_df.columns) > 2:
                        semantic_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                                                          'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')

                        features = get_columns_of_type(nodes_df, semantic_types).values.astype(np.float32)
                        
                        
                        if tensor:
                                features = tf.convert_to_tensor(features)
                                to_return= tf.concat([features, node_id], -1)
                        else:
                                to_return=np.concatenate([features, node_id], axis = -1)
                else:
                        to_return = node_id


                return to_return

        def _make_input_features(self, nodes_df, tensor = False, num_nodes = None):# tensor = True):
                num_nodes = num_nodes if num_nodes is not None else nodes_df.shape[0]

                if tensor:
                        node_id = tf.cast(tf.eye(num_nodes), dtype = tf.float32)
                        #node_id = tf.sparse.eye(nodes_df.shape[0])
                else:
                        #node_id = scipy.sparse.identity(nodes_df.shape[0], dtype = np.float32) #
                        node_id = np.eye(num_nodes)

                self._input_columns = num_nodes
                # preprocess features, e.g. if non-numeric / text?
                if len(nodes_df.columns) > 2:
                        semantic_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                                                'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')


                        try:
                                features = get_columns_of_type(nodes_df, semantic_types)
                        except:
                                features = nodes_df
                        features = features[[c for c in features.columns if 'label' not in c and 'nodeID' not in c and 'index' not in c.lower()]]
                        features = features.values.astype(np.float32)

                        #features = features[:, 1:]
                        #import IPython; IPython.embed()
                        self._input_columns += features.shape[-1] 


                        if tensor:
                                features = tf.convert_to_tensor(features)
                                #features = tf.contrib.layers.dense_to_sparse(tf.convert_to_tensor(features))
                                to_return= tf.concat([node_id, features], -1)
                        else:
                                to_return=np.concatenate([node_id, features], axis = -1)
                        
                else:
                        to_return = node_id
                        
                        
                if tensor:
                        return to_return #tf.contrib.layers.dense_to_sparse(to_return)
                else:
                        return to_return#csr_matrix(to_return)
                #if self._input is None:
                #self._input = keras.layers.Input(tensor = to_return if tensor else tf.convert_to_tensor(to_return))
                #else:
                #        tf.assign(self._input, to_return if tensor else tf.convert_to_tensor(to_return))



        def _pred(self, adj_input, feature_input, embedding = False, _slice = False):
                if isinstance(adj_input, list):
                        fd = {self.adj_inds: adj_input[0],
                                     self.adj_vals: adj_input[-1],
                                     self.feature_input: feature_input}
  
                   
                else:
                        fd = {self.adj_input: adj_input,       
                              self.feature_input: feature_input}#,
                
                if not embedding:
                        if not _slice:
                                output = self.sess.run([self.y_pred], feed_dict = fd)
                        else:
                                output = self.sess.run([self.y_pred_slice], feed_dict = fd)
                else:
                        if not _slice:
                                output = self.sess.run([self.embedding], feed_dict = fd)
                        else:
                                output = self.sess.run([self.embedding_slice], feed_dict = fd)
                return output


        def fit(self, *, timeout : float = None, iterations : int = None) -> None:

                
                if self.fitted:
                        return CallResult(None, True, 1)

                # ******************************************
                # Feel free to fill in less important hyperparameters or example architectures here
                # ******************************************
                layers = []
                self.keras_fit = False
                self._task = 'classification'
                self._act = 'relu'
                self._epochs = self.hyperparams['epochs'] #200 if self._num_training_nodes < 10000 else 50
                self._units = [self.hyperparams['layer_size']]*self.hyperparams['layers'] 
                self._mix_hops = self.hyperparams['adjacency_order']
                self._modes = 1 # TO DO : INFER FROM DATA FOR LINK PREDICTION
                try:
                        self._lr = self.hyperparams['lr']
                except:
                        self._lr = 0.01
                #self._optimizer = keras.optimizers.Adam(self._lr)
                self._extra_fc = self.hyperparams['layer_size']
                # self._adj and self._input already set as keras Input tensors
 

                #odes = None, units = [100, 100], mix_hops = 2, modes = 1, lr = 0.0005, extra_fc = 100, 
                #        outputs_tensor = None, inds_tensor = None, sparse = True, 
                #        labeled = None, input_columns = None, label_unique = None
                use_network = True
                if use_network:
                        self.network = GCN_Network(nodes = self._num_training_nodes, units = self._units, mix_hops = self._mix_hops, 
                                lr = self._lr, extra_fc = self._extra_fc, modes = self._modes, epochs = self._epochs,
                                outputs_tensor = self.outputs_tensor, inds_tensor = self.inds_tensor, logger = _logger,
                                input_columns = self._input_columns, label_unique = self._label_unique, sparse = True)


                        self.network.fit(self._adj, self._input)
                else:
                        #adj_input = keras.layers.Input(shape = self._adj.shape[1:], name = 'adjacency', sparse = True, dtype = tf.float32)
                        inp_tensors = False
                        if inp_tensors:
                                adj_input = keras.layers.Input(tensor = self._adj, name = 'adjacency', dtype = 'float32')
                                feature_input = keras.layers.Input(tensor = self._input, name = 'features', dtype = 'float32')
                                # previous 
                                #adj_input = keras.layers.Input(tensor = self._adj, batch_shape = (None, tf.shape(self._adj)[-1]), name = 'adjacency', dtype = tf.float32)
                                #feature_input = keras.layers.Input(tensor = self._input, batch_shape = (None, self._input_columns), name = 'features', dtype = tf.float32)
                        else:
                                adj_input = keras.layers.Input(shape = (self._num_training_nodes,), name = 'adjacency', dtype = 'float32')#, sparse = True)
                                feature_input = keras.layers.Input(shape = (self._input_columns,), name = 'features', dtype = 'float32')#, sparse =True)
                        

                        self.y_true = keras.layers.Input(tensor = self.outputs_tensor, name = 'y_true', dtype = 'float32')
                        self.inds = keras.layers.Input(tensor = self.inds_tensor, dtype='int32', name = 'training_inds')

                        self.layers.append(self.y_true)
                        self.layers.append(self.inds)

                        if self.sparse:
                                self.adj_inds = tf.placeholder(tf.int64)
                                self.adj_vals = tf.placeholder(tf.float32)
                                self.adj_input = tf.SparseTensor(self.adj_inds, self.adj_vals, dense_shape = (self._num_training_nodes, self._num_training_nodes))
                        else:
                                self.adj_input = tf.sparse_placeholder(tf.float32, shape = (self._num_training_nodes, self._num_training_nodes))
                        print()
                        print("SELF NUM TRAINING NODES ", self._num_training_nodes)
                        print()

                        self.feature_input = tf.placeholder(tf.float32, shape = (self._num_training_nodes, self._input_columns)) #tf.sparse_placeholder(tf.float32, shape = (self._num_training_nodes, self._num_training_nodes))

                        # using self.y_true and self.inds right now...
                        #self.y = tf.placeholder(tf.float32, [None, self.training_outputs.shape[-1]], name='y')
                        #self.ph_indices = tf.placeholder(tf.int64, [None])
                        

                        self.layers.append(self.adj_inds)
                        self.layers.append(self.adj_vals)
                        self.layers.append(self.adj_input)
                        self.layers.append(self.feature_input)
                        #y_true = keras.layers.Input(shape = (self.training_outputs.shape[-1],), name = 'y_true')
                        #inds = keras.layers.Input(shape = (None,), dtype=tf.int32, name = 'training_inds')
                        

                        #feature_input = keras.layers.Input(shape = self._input.shape[1:], name = 'features')#sparse =True) #tensor =  if tensor else tf.convert_to_tensor(to_return))
                        self.embedding = self._make_gcn(self.adj_input, self.feature_input,
                                h_dims = self._units,
                                mix_hops = self._mix_hops, 
                                modes = self._modes)

                        self.layers.append(self.embedding)
                        #self._embedding_model = keras.models.Model(inputs = self._input, outputs = embedding)
                        # make_task
                        # ********************** ONLY NODE CLF RIGHT NOW *******************************
                        #if self._task == 'node_clf':
                        if self._extra_fc is not None:
                                self.embedding = tf.layers.Dense(self._extra_fc, activation = self._act)(self.embedding)
                                self.layers.append(self.embedding)

                        label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
                        self.y_pred = tf.layers.Dense(self._label_unique, activation = label_act, name = 'y_pred')(self.embedding)
                        
                        self.layers.append(self.y_pred)
                        def semi_supervised_slice(inputs, first = None):
                                if isinstance(inputs, list):
                                        tensor = inputs[0]
                                        inds = inputs[-1]
                                        inds = tf.squeeze(inds) 
                                else:  
                                        tensor = inputs
                                        inds = np.arange(first, dtype = np.float32)
                                try:
                                        sliced = tf.gather(tensor, inds, axis = 0)
                                except:
                                        sliced = tf.gather(tensor, tf.cast(inds, tf.int32), axis = 0)
                                #sliced.set_shape([None, sliced.get_shape()[-1]])
                                #return tf.cast(sliced, tf.float32)
                                return tf.cast(tf.reshape(sliced, [-1, tf.shape(tensor)[-1]]), tf.float32)

                        # def semi_supervised_slice(tensor, inds):
                        #         sliced = tf.gather(tensor, inds, axis = 0)
                        #         #sliced.set_shape([None, sliced.get_shape()[-1]])
                        #         return sliced
             

                        outputs = []
                        loss_functions = []
                        loss_weights = []
                
                        #outputs.append(y_pred)

                        if label_act == 'softmax':  # if self._task == 'node_clf': 
                                #loss_functions.append(keras.objectives.categorical_crossentropy)
                                loss_function = keras.objectives.categorical_crossentropy
                        else: #f label_act == 'sigmoid':
                                #loss_functions.append(keras.objectives.binary_crossentropy)
                                loss_function = keras.objectives.binary_crossentropy
                        #    loss_functions.append(keras.objectives.mean_squared_error)#mse                                            


                        # self.embedding_slice = semi_supervised_slice([self.embedding, self.inds])
                        # self.y_pred_slice = semi_supervised_slice([self.y_pred, self.inds])
                        # self.y_true_slice = semi_supervised_slice([self.y_true, self.inds])
                        # self.slice_loss = keras.layers.Lambda(loss_fun, arguments = {'function': loss_function, 'first': self._num_labeled_nodes})([self.y_true_slice, self.y_pred_slice])
          
                        self.embedding_slice = keras.layers.Lambda(semi_supervised_slice)([self.embedding, self.inds])
                        # Note: Y-true is an input tensor
                        self.y_pred_slice = keras.layers.Lambda(semi_supervised_slice)([self.y_pred, self.inds])#, arguments = {'inds': self.training_inds})(y_pred)
                        # doesn't acutally use total / keep
                        self.y_true_slice = keras.layers.Lambda(semi_supervised_slice)([self.y_true, self.inds])
                        
                        self.slice_loss = keras.layers.Lambda(loss_fun, arguments = {'function': loss_function, 'first': self._num_labeled_nodes})([self.y_true_slice, self.y_pred_slice])
                        
                        full_loss = keras.layers.Lambda(assign_scattered)([self.slice_loss, self.y_pred, self.inds])
                        #full_loss = keras.layers.Lambda(dummy_concat, arguments = {'total': self._num_training_nodes, 'keep':self._num_labeled_nodes})([outputs, y_pred])
                        
                        #y_pred_full = keras.layers.Lambda(dummy_concat, arguments = {'total': self._num_training_nodes, 'keep': self._num_labeled_nodes})([y_pred_slice, y_pred])
                        
                        #y_true_slice = keras.layers.Lambda(semi_supervised_slice, arguments = {'first': self._num_labeled_nodes})(y_true)
                        #                
                        #outputs.append(y_pred_full)
                        #loss_functions.append(loss_function)
                        
                        #loss_functions = [identity]
                        
                        #outputs = loss_functions(y_true, y_pred)
                        #keras.layers.Lambda(loss_functions)([y_true, y_pred])

                        outputs.append(full_loss)
                        loss_functions.append(identity)
                        loss_weights.append(1.0)
                


                        tf_loss = self.slice_loss #full_loss

                        learn_rate = tf.placeholder(tf.float32, [], 'learn_rate')
                        optimizer = tf.train.MomentumOptimizer(learn_rate, 0.7, use_nesterov=True)
                        self.train_op = slim.learning.create_train_op(
                                tf_loss, optimizer, gradient_multipliers=[])

                        LAST_STEP = collections.Counter()
                        
                        #self.saver = tf.train.Saver() 
                        self.sess = tf.Session()
                        self.sess.run(tf.global_variables_initializer())

                        def step(lr=None, columns=None):
                            i = LAST_STEP['step']
                            LAST_STEP['step'] += 1
                            #feed_dict[is_training] = True
                            #feed_dict[ph_indices] = train_indices

                            if not isinstance(self._adj, list):
                                    feed_dict = {self.adj_input: self._adj,       
                                                 self.feature_input: self._input}#,
                            else:
                                    feed_dict = {self.adj_inds: self._adj[0],
                                                 self.adj_vals: self._adj[-1],
                                                 self.feature_input: self._input}
                                                 
                                #y: self.training_outputs
                                #}
                            if lr is not None:
                              feed_dict[learn_rate] = lr
                            
                                # Train step
                            #train_preds, loss_value, _ = sess.run((sliced_output, label_loss, train_op), feed_dict)


                            train_preds, loss_value, _ = self.sess.run((self.y_pred_slice, self.slice_loss, self.train_op), feed_dict)

                            if np.isnan(loss_value).any():
                                    print('NaN value reached. Debug please.')
                                    import IPython; IPython.embed()

                            return train_preds, loss_value
                                

                        # STOPPING AFTER CERTAIN AMOUNT OF TIME?
                        tic = time.time()
                        for i in range(self._epochs):
                                # TO DO : anneal learning rate
                                preds, loss_value = step(self._lr)
                                print("Epoch ", i, " Loss ", np.mean(loss_value))
                                if time.time()-tic > 3000:
                                        break
                        # READY TO DELETE ALL OF THIS


                # KERAS FITTING 
                if False: #self.keras_fit:
                        # fit keras
                        self.model = keras.models.Model(inputs = [adj_input, feature_input, self.y_true, self.inds], outputs = outputs)#, feature_input], outputs = outputs)
                        self.pred_model = keras.models.Model(inputs =  [adj_input, feature_input, self.inds], outputs = [y_pred_slice])#, feature_input], outputs = outputs)
                        self.embedding_model = keras.models.Model(inputs = [adj_input, feature_input], outputs = [embedding])#, feature_input], outputs = outputs)
                        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)

                      
                        # self.model.fit(x = [np.squeeze(self.training_outputs), np.squeeze(self.training_inds)], #[self._adj],#, self._input], # already specified as tensors
                        #                y = [np.squeeze(self.training_outputs)],# + [np.squeeze(self.training_outputs)],                               
                        #                shuffle = False, epochs = self._epochs, 
                        #                batch_size = self._num_training_nodes) #self.training_inds.shape[0])
                        #                #batch_size = self._num_training_nodes) 

                        self.model.fit(x = [self._adj, self._input], #[self._adj],#, self._input], # already specified as tensors
                                       y = [self.training_outputs],# + [np.squeeze(self.training_outputs)],                               
                                       shuffle = False, epochs = self._epochs, 
                                       batch_size = self._num_training_nodes,
                                       verbose = 0
                        ) #self.training_inds.shape[0])
                                       #batch_size = self._num_training_nodes) 
                        
                        # all must have same # of samples
                        # self.model.fit(x = [self.training_outputs, self.training_inds, self._adj, self._input], #[self._adj],#, self._input], # already specified as tensors
                        #                y = [self.training_outputs],# + [np.squeeze(self.training_outputs)],                               
                        #                shuffle = False, epochs = self._epochs, 
                        #                batch_size = self._num_training_nodes) #self.training_inds.shape[0])

                self.fitted = True


                make_keras_pickleable()
                return CallResult(None, True, 1)
        
        def _parse_inputs(self, inputs : Input):
                if len(inputs) == 3:
                        learning_df = inputs[0]
                        nodes_df = inputs[1]
                        edges_df = inputs[-1]
                elif len(inputs) == 2:
                        learning_df = None
                        nodes_df = inputs[1]
                        edges_df = inputs[-1]
                else:
                        print("********** GCN INPUTS ***********", inputs)
                        raise ValueError("Check inputs to GCN")

                return learning_df, nodes_df, edges_df

                                                 
        def produce(self, *, inputs : Input, outputs : Output, timeout : float = None, iterations : int = None) -> CallResult[Output]:
                make_keras_pickleable()
                if self.fitted:

                        # embed ALL (even unlabelled examples)
                        learning_df, nodes_df, edges_df = self._parse_inputs(inputs)
                        _adj, _input, targets = self._get_training_data(learning_df, nodes_df, edges_df)
                        # sets inds, target tensors
                        self._set_training_values(learning_df, targets)


                        print("PRODUCE ", learning_df)
                        print("NODES ", nodes_df)

                        if False:#self.keras_fit:
                                try:
                                        result = self.pred_model.predict([_adj.todense(), _input.todense()], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                                except Exception as e:
                                        print(type(self.training_inds), self.training_inds.shape, np.squeeze(self.training_inds).shape)
                                        #result = self.pred_model.predict([np.squeeze(self.training_inds), _adj.todense(), _input.todense()], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                                        result = self.pred_model.predict([_adj.todense(), _input.todense(),np.squeeze(self.training_inds)[:]], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                                        #self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0]) #, node_subset = node_subset.values.astype(np.int32))
                                        #self._input = self._make_input_features(nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)])
                                        #result = self.model.predict([_adj.todense(), _input.todense()], steps = 1)

                        else:
                                result = self.network.pred(_adj, _input, embedding = False, _slice = True)
                                #result = self._pred(_adj, _input, embedding = False, _slice = True)
                                
                        result = np.argmax(result, axis = -1) #if not self.hyperparams['return_embedding'] else result

                        if self.label_encode is not None:
                                result = self.label_encode.inverse_transform(result)
        
                        # line graph produce doesn't have to give a single prediction! can return softmax               
                        if self.hyperparams['return_embedding']:
                                # try:
                                #         output_embed = self.model.layers[-2].output 
                                #         func = K.function([self.model.input[0], self.model.input[1], K.learning_phase()], [output_embed])
                                #         embed = func([adj, inp, 1.])[0]
                                # except:
                                if self.keras_fit:
                                        try:
                                                embed = self.embedding_model.predict([_adj.todense(), _input.todense()], steps = 1)
                                        except:
                                                embed = self.embedding_model.predict([self.training_inds, _adj.todense(), _input.todense()], steps = 1)
                                                
                                        embed = embed[self.training_inds]
                                else:
                                        embed = self.network.pred(_adj, _input, embedding = True, _slice = True)
                                
                                try:
                                        result = np.concatenate([result, embed], axis = -1)
                                except:
                                        result = np.concatenate([np.expand_dims(result, 1), embed], axis = -1)


                else:
                        raise Error("Please call fit first")
                
                # ******************************************
                # Subroutine to get output in proper D3M format

                # ** Please confirm / double check **
                # ******************************************


                #if self.hyperparams['return_list']:
                #        result_np = container.ndarray(result, generate_metadata = True)
                #        return_list = d3m_List([result_np, inputs[1], inputs[2]], generate_metadata = True)        
                #        return CallResult(return_list, True, 1)
                


                # INVERSE TRANSFORM ON NODE_ENCODE???
                if not self.hyperparams['return_embedding']:
                        output = d3m_DataFrame(result, index = learning_df['d3mIndex'], columns = [learning_df.columns[-1]], generate_metadata = True, source = self)
                else:
                        output = d3m_DataFrame(result, index = learning_df['d3mIndex'], generate_metadata = True, source = self)                     
                        
                #output.index.name = 'd3mIndex'
                output.index = learning_df.index.copy()
                outputs = output
                
                self._training_indices = [c for c in learning_df.columns if isinstance(c, str) and 'index' in c.lower()]

                output = utils.combine_columns(return_result='new', #self.hyperparams['return_result'],
                                               add_index_columns=True,#self.hyperparams['add_index_columns'], 
                                               inputs=learning_df, columns_list=[output], source=self, column_indices=self._training_indices)
               
                return CallResult(output, True, 1)
                #return CallResult(outputs, True, 1)
                        
                
                # TO DO : continue_fit, timeout
                
        def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, outputs : Output, timeout: float = None, iterations: int = None) -> MultiCallResult:
                return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, outputs=outputs)

        def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, outputs : Output, timeout : float = None, iterations : int = None) -> MultiCallResult:
                return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, outputs=outputs)

        def get_params(self) -> GCN_Params:

                # fill in with model attributes 

                return GCN_Params(
                        fitted = self.fitted,
                        network = self.network,
                        label_encode = self.label_encode,
                        node_encode = self.node_encode)#,
                        #model = self.model,
                        #pred_model = self.pred_model,
                        #embed_model = self.embedding_model,
                        #weights = self.model.get_weights(),
                        #pred_weights = self.pred_model.get_weights(),
                        #embed_weights = self.embedding_model.get_weights(),
                        #adj = self._adj)
        
        def set_params(self, *, params: GCN_Params) -> None:

                # assign model attributes (e.g. in loading from pickle)

                self.fitted = params['fitted']
                self.network = params['network']
                self.node_encode = params['node_encode']
                self.label_encode = params['label_encode']
                #self.model = params['model']
                #self.model.set_weights(params['weights'])
                #self.pred_model = params['pred_model']
                #self.pred_model.set_weights(params['pred_weights'])
                #self.embedding_model = params['embed_model']
                #self.embedding_model.set_weights(params['embed_weights'])
                #self._adj = params['adj']

        def _get_ph(self):
                # num supports?  
                num_supports = 1
                self._adj = [keras.layers.Input(tensor = tf.sparse_placeholder(tf.float32), name = 'support_'+str(i)) for i in range(num_supports)]
                #self._adj = self._adj[0] if len(self._adj) == 0 else self._adj

                # input:  vector of features/ convolved with Identity on nodes
                self._input = tf.sparse_placeholder(tf.float32) #, shape=tf.constant(features[2], dtype=tf.int64)) # why is this integer?

                return self._adj
        #     placeholders = {
        #     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        #     
                
                #else:
                #        x = inputs[0] if isinstance(inputs, list) else inputs
                #        adj = self._adj
                        # exponent for mixhop  
        
                #x = inputs[0] if isinstance(inputs, list) else inputs
                # exponent for mixhop  
                #return sparse_exp_ax(adj, x, exponent = k)      

        def _make_gcn(self, adj, features, h_dims = [100, 100], mix_hops = 5, modes = 1):
                a = adj
                x = features #self._input
                #print('adj ', adj)
                #print('features ', features)
                # where should number of modes be handled?
                for h_i in range(len(h_dims)):
                        act_k = []

                        for mode in range(modes):
                                for k in range(mix_hops+1):
                                        #pre_w = keras.layers.Lambda(gcn_layer, arguments = {'k': k})([x, self._adj])
                                        #pre_w = tf.sparse.to_dense(sparse_exp_ax(a, x, exponent = k))
                                        #pre_w = tf.sparse_tensor_to_dense(sparse_exp_ax(a, x, exponent = k))
                                        pre_w = sparse_exp_ax(a, x, exponent = k)
                                        self.layers.append(pre_w)
                                        #print("PRE W ")
                                        #print(pre_w)
                                        #pre_w = GCN_Layer(k = k)([x,a])#, adj = a)(x)
                                        #import IPython; IPython.embed()
                                        #keras.layers.Lambda(function, arguments = {'k': k})(x)
                                        act = tf.layers.Dense(h_dims[h_i], activation = self._act, name='w'+str(k)+'_'+str(mode)+'_'+str(h_i))(pre_w)
                                        act_k.append(act)
                                        self.layers.append(act)
                        x = tf.concat(act_k, name = 'mix_'+str(mix_hops)+'hops_'+str(h_i), axis = -1) #keras.layers.Concatenate(axis = -1, name = 'mix_'+str(mix_hops)+'hops_'+str(h_i))(act_k)
                        self.layers.append(x)
                # embedding tensor (concatenation of k)
                return x


         
