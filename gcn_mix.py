import os
import sys
import typing
import networkx
import numpy as np
import pdb

import tensorflow as tf
import keras
import pandas as pd
import copy 
import importlib

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
                                 
                                        
class GCN_Params(params.Params):

        ''' 
        Attributes necessary to resume training or run on test data (if loaded from pickle)

        Code specifications of parameters: 
                https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/params.py
        '''

        fitted: typing.Union[bool, None] # fitted required, set once primitive is trained
        model: keras.models.Model #typing.Union[keras.models.Model, None]d
        pred_model: keras.models.Model
        embed_model: keras.models.Model
        weights: typing.Union[typing.Any, None]
        pred_weights: typing.Union[typing.Any, None]
        embed_weights: typing.Union[typing.Any, None]
        adj: typing.Union[tf.Tensor, tf.SparseTensor, tf.Variable, keras.layers.Input, np.ndarray, csr_matrix, None]

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
                lower = 1,
                upper = 500,
                default = 100,
                #q = 5e-8,                                                                                                                                                                 
                description = 'number of epochs to train',
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

# all primitives must be pickle-able, and this should do the trick for Keras models

def loss_fun(inputs, function = None, first = None):
        if isinstance(function, str):
                import importlib
                mod = importlib.import_module('keras.objectives')
                function = getattr(mod, function)
        try:
                return function(inputs[0], inputs[-1]) if function is not None else inputs
        except:
                inputs[0] = tf.gather(inputs[0], np.arange(first))
                return function(inputs[0], inputs[-1]) if function is not None else inputs



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

        return df.select_columns(columns_to_use)

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
                
                return sparse_exp_ax(self.adj, x, exponent = self.k)
                
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
                "python_path": "d3m.primitives.feature_construction.gcn_mixhop.DSBOX",
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
                


                if self.hyperparams['line_graph']:
                        # try:
                        #         #idx = edges_df['d3mIndex']
                        #         my_edges = edges_df.join(learning_df, on ='d3mIndex', how='inner')
                        #         #edges_df = edges_df.loc[learning_df['d3mIndex'].astype(np.int32)] #
                        # except Exception as e:
                        #         print()
                        #         print("*"*500)
                        #         print("edges indexing error ", e)
                        #         print("*"*500)
                        #         try:
                        #                edges_df = edges_df.astype(object)
                        #                #pdb.set_trace()
                        #                my_edges = pd.merge(edges_df.assign(x=edges_df.source.astype(int)), 
                        #                                    learning_df.assign(x=learning_df.source_nodeID.astype(int)), 
                        #                                    how = 'right', 
                        #                                    left_on = ['source'],#, 'target'],
                        #                                    right_on = ['source_nodeID'])#, 'target_nodeID'])
                        #                print()
                        #                print(my_edges)
                        #                print()
                        #                my_edges.set_index('d3mIndex')
                        #         except Exception as e:
                        #                 print()
                        #                 print("MERGING EXCEPTION ", e)
                        #                 print()
                        # print(my_edges)
                        #edges_df = my_edges
                        self._num_training_nodes = edges_df.values.shape[0]
                        self._adj = self._make_line_adj(edges_df)
                        self._input = self._make_line_inputs(edges_df)
                else:

                        #nodes_df = nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]
                        #node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]

                        try:
                                self._num_training_nodes = node_subset.values.shape[0]
                        except:
                                self._num_training_nodes = nodes_df.values.shape[0]
                        #self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0], tensor = True)#, node_subset = node_subset.values.astype(np.int32))
                        #self._input = self._make_input_features(nodes_df, tensor = True)#.loc[learning_df['d3mIndex'].astype(np.int32)])#.index])
                
                        self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0])#, node_subset = node_subset.values.astype(np.int32))
                        self._input = self._make_input_features(nodes_df)#.loc[learning_df['d3mIndex'].astype(np.int32)])#.index])

                # dealing with outputs
                #if self._task in ['clf', 'class', 'classification', 'node_clf']:

                target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                use_outputs = False
                if use_outputs:
                        targets =  get_columns_of_type(outputs, target_types)
                else:
                        targets =  get_columns_of_type(learning_df, target_types)
                

                
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

                self._set_training_values(learning_df, targets)
                
                self.fitted = False



        def _set_training_values(self, learning_df, targets):
                self.training_inds = learning_df['d3mIndex'].astype(np.int32).values

                self._label_unique = np.unique(targets.values).shape[0]
                #self._label_unique = np.unique(targets).shape[0]
                try:
                        self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                except:
                        self.label_encode = LabelEncoder()
                        self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                
                #self.training_outputs = to_categorical(self.label_encode.fit_transform(targets), num_classes = np.unique(targets).shape[0])
                
                self._num_labeled_nodes = self.training_outputs.shape[0]
                training_outputs = np.zeros(shape = (self._num_training_nodes, self.training_outputs.shape[-1]))
                for i in range(self.training_inds.shape[0]):
                        training_outputs[self.training_inds[i],:]= self.training_outputs[i, :]
                self.training_outputs = training_outputs

                self.outputs_tensor = tf.constant(self.training_outputs)
                self.inds_tensor = tf.constant(np.squeeze(self.training_inds), dtype = tf.int32)

                self.y_true = keras.layers.Input(tensor = self.outputs_tensor, name = 'y_true', dtype = 'float32')
                self.inds = keras.layers.Input(tensor = self.inds_tensor, dtype='int32', name = 'training_inds')

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
                
                return sources, dests

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

                
        def _make_adjacency(self, edges_df, num_nodes = None, tensor = False, #True, 
                            node_subset = None):
                
                sources, dests = self._get_source_dest(edges_df)

                #attr_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                #                          'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')
                #attrs = get_columns_of_type(edges_df, attr_types)
                
                #attrs = self.node_enc.transform(attrs.value)

                self.node_enc = LabelEncoder()
                #id_col = [i for i in nodes_df.columns if 'node' in i and 'id' in i.lower()][0]
                to_fit = node_subset if node_subset is not None else np.concatenate([sources.values,dests.values], axis = -1).ravel()

                self.node_enc.fit(to_fit) #nodes_df[id_col].values)


                if node_subset is not None:
                        node_subset = node_subset.values if isinstance(node_subset,pd.DataFrame) else node_subset 
                        num_nodes = node_subset.shape[0]

                        inds = [i for i in sources.index if sources.loc[i, sources.columns[0]] in node_subset and dests.loc[i, dests.columns[0]] in node_subset]

                        
                        sources = sources.loc[inds] 
                        dests = dests.loc[inds]


                        #attrs = attrs.loc[inds]
                
                sources[sources.columns[0]] = self.node_enc.transform(sources.values)
                dests[dests.columns[0]] = self.node_enc.transform(dests.values)
                
                # accomodate weighted graphs ??
                if tensor:
                        adj = tf.SparseTensor([[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])], [1.0 for i in range(sources.values.shape[0])], dense_shape = (num_nodes, num_nodes))
                else:
                        adj = csr_matrix(([1.0 for i in range(sources.values.shape[0])], ([sources.values[i, 0] for i in range(sources.values.shape[0])], [dests.values[i,0] for i in range(sources.values.shape[0])])), shape = (num_nodes, num_nodes), dtype = np.float32)
                #tf.sparse.placeholder(dtype,shape=None,name=None)
        
                return adj
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

                        features = get_columns_of_type(nodes_df, semantic_types).values.astype(np.float32)
                        self._input_columns += features.shape[-1]

                        if tensor:
                                features = tf.convert_to_tensor(features)
                                to_return= tf.concat([features, node_id], -1)
                        else:
                                to_return=np.concatenate([features, node_id], axis = -1)
                        
                else:
                        to_return = node_id
                        


                return csr_matrix(to_return)
                #if self._input is None:
                #self._input = keras.layers.Input(tensor = to_return if tensor else tf.convert_to_tensor(to_return))
                #else:
                #        tf.assign(self._input, to_return if tensor else tf.convert_to_tensor(to_return))


        def fit(self, *, timeout : float = None, iterations : int = None) -> None:

                
                if self.fitted:
                        return CallResult(None, True, 1)

                # ******************************************
                # Feel free to fill in less important hyperparameters or example architectures here
                # ******************************************
                self._task = 'classification'
                self._act = 'relu'
                self._epochs = 200 if self._num_training_nodes < 10000 else 50
                self._units = [100, 100]
                self._mix_hops = self.hyperparams['adjacency_order']
                self._modes = 1
                self._lr = 0.0005
                self._optimizer = keras.optimizers.Adam(self._lr)
                self._extra_fc = 100 
                # self._adj and self._input already set as keras Input tensors
 
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

                #y_true = keras.layers.Input(shape = (self.training_outputs.shape[-1],), name = 'y_true')
                #inds = keras.layers.Input(shape = (None,), dtype=tf.int32, name = 'training_inds')
                
                #feature_input = keras.layers.Input(shape = self._input.shape[1:], name = 'features')#sparse =True) #tensor =  if tensor else tf.convert_to_tensor(to_return))
                embedding = self._make_gcn(adj_input, feature_input,
                        h_dims = self._units,
                        mix_hops = self._mix_hops, 
                        modes = self._modes)

                #self._embedding_model = keras.models.Model(inputs = self._input, outputs = embedding)
                # make_task
                # ********************** ONLY NODE CLF RIGHT NOW *******************************
                #if self._task == 'node_clf':
                if self._extra_fc is not None:
                        embedding = keras.layers.Dense(self._extra_fc, activation = self._act)(embedding)

                label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
                y_pred = keras.layers.Dense(self._label_unique, activation = label_act, name = 'y_pred')(embedding)

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


                
                # Note: Y-true is an input tensor
                y_pred_slice = keras.layers.Lambda(semi_supervised_slice)([y_pred, self.inds])#, arguments = {'inds': self.training_inds})(y_pred)
                # doesn't acutally use total / keep
                
                y_true_slice = keras.layers.Lambda(semi_supervised_slice)([self.y_true, self.inds])

                slice_loss = keras.layers.Lambda(loss_fun, arguments = {'function': loss_function, 'first': self._num_labeled_nodes})([y_true_slice, y_pred_slice])
                


                full_loss = keras.layers.Lambda(assign_scattered)([slice_loss, y_pred, self.inds])
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
        
                # 
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
                        if not self.hyperparams['line_graph']:
                                ##node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]
                                ##self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0], node_subset = node_subset.values.astype(np.int32))
                                ##self._input = self._make_input_features(nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]) #.index])
                                #nodes_df = nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]
                                ##node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]
                                ##self._num_training_nodes = node_subset.values.shape[0]
                                #adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0], node_subset = node_subset.values.astype(np.int32))
                                #inp = self._make_input_features(nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)])
                                

                                try:
                                        self._num_training_nodes = node_subset.values.shape[0]
                                except:
                                        self._num_training_nodes = nodes_df.values.shape[0]
                                _adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0])#, node_subset = node_subset.values.astype(np.int32))
                                _input = self._make_input_features(nodes_df)#.loc[learning_df['d3mIndex'].astype(np.int32)])#.index])
                                

                        target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                    
                        targets =  get_columns_of_type(learning_df, target_types)
                        
                        self._set_training_values(learning_df, targets)


                        #try:
                        #        output_pred = self.model.layers[-1].output # all layer outputs
                        #        func = K.function([self.model.input[0], self.model.input[1], K.learning_phase()], [output_pred])
                        #        
                         #       result = func([adj, inp, 1.])[0]
                        #except: # could be preferable, but both prediction methods should work
                        try:
                                result = self.pred_model.predict([_adj.todense(), _input.todense()], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                        except Exception as e:
                                print(type(self.training_inds), self.training_inds.shape, np.squeeze(self.training_inds).shape)
                                print("list ", np.array(list(np.squeeze(self.training_inds))).shape)
                                print('INDS TENSOR ', self.inds)
                                #result = self.pred_model.predict([np.squeeze(self.training_inds), _adj.todense(), _input.todense()], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                                result = self.pred_model.predict([_adj.todense(), _input.todense(),np.squeeze(self.training_inds)[:]], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                                #self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0]) #, node_subset = node_subset.values.astype(np.int32))
                                #self._input = self._make_input_features(nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)])
                                #result = self.model.predict([_adj.todense(), _input.todense()], steps = 1)


                        
                        result = np.argmax(result, axis = -1) #if not self.hyperparams['return_embedding'] else result
                       
                        if self.hyperparams['return_embedding']:
                                # try:
                                #         output_embed = self.model.layers[-2].output 
                                #         func = K.function([self.model.input[0], self.model.input[1], K.learning_phase()], [output_embed])
                                #         embed = func([adj, inp, 1.])[0]
                                # except:
                                try:
                                        embed = self.embedding_model.predict([_adj.todense(), _input.todense()], steps = 1)
                                except:
                                        embed = self.embedding_model.predict([self.training_inds, _adj.todense(), _input.todense()], steps = 1)
                                        
                                embed = embed[self.training_inds]
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
                        model = self.model,
                        pred_model = self.pred_model,
                        embed_model = self.embedding_model,
                        weights = self.model.get_weights(),
                        pred_weights = self.pred_model.get_weights(),
                        embed_weights = self.embedding_model.get_weights(),
                        adj = self._adj)
        
        def set_params(self, *, params: GCN_Params) -> None:

                # assign model attributes (e.g. in loading from pickle)

                self.fitted = params['fitted']
                self.model = params['model']
                self.model.set_weights(params['weights'])
                self.pred_model = params['pred_model']
                self.pred_model.set_weights(params['pred_weights'])
                self.embedding_model = params['embed_model']
                self.embedding_model.set_weights(params['embed_weights'])
                self._adj = params['adj']

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
                
                # where should number of modes be handled?
                for h_i in range(len(h_dims)):
                        act_k = []
                        for k in range(mix_hops+1):
                                #pre_w = keras.layers.Lambda(gcn_layer, arguments = {'k': k})([x, self._adj])
                                pre_w = GCN_Layer(k = k)([x,a])#, adj = a)(x)
                                #keras.layers.Lambda(function, arguments = {'k': k})(x)
                                act = keras.layers.Dense(h_dims[h_i], activation = self._act, name='w'+str(k)+'_'+str(h_i))(pre_w)
                                act_k.append(act)
                        x = keras.layers.Concatenate(axis = -1, name = 'mix_'+str(mix_hops)+'hops_'+str(h_i))(act_k)

                # embedding tensor (concatenation of k)
                return x


         
