import os
import sys
import typing
import networkx
import numpy as np

import tensorflow as tf
import keras
import pandas as pd
import copy 

import keras.objectives
import keras.backend as K
#from sklearn import preprocessing
import tempfile
import scipy.sparse
from scipy.sparse import csr_matrix
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

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



Input = typing.Union[container.List, container.DataFrame]
Output = container.DataFrame
                                 
                                        
class GCN_Params(params.Params):

        ''' 
        Attributes necessary to resume training or run on test data (if loaded from pickle)

        Code specifications of parameters: 
                https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/params.py
        '''

        fitted: typing.Union[bool, None] # fitted required, set once primitive is trained
        model: typing.Union[keras.models.Model, None]
        adj: typing.Union[tf.Tensor, tf.SparseTensor, tf.Variable, keras.layers.Input, np.ndarray, csr_matrix, None]
        weights: typing.Union[typing.Any, None]

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
        beta = UniformInt( 
                lower = 1,
                upper = 20,
                default = 1,
                #q = 1,
                description = 'seen edge reconstruction weight',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )
        alpha = Uniform(
                lower = 1e-8,
                upper = 1,
                default = 1e-5,
                #q = 5e-8,
                description = 'first order proximity weight',
                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )
        return_embedding = UniformBool(
                default = True,
                description='return embedding as features alongside classification prediction',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )


# all primitives must be pickle-able, and this should do the trick for Keras models



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
                else:
                        x = inputs

                return sparse_exp_ax(self.adj, x, exponent = self.k)
                
        def compute_output_shape(self, input_shape):
                return input_shape if not isinstance(input_shape, list) else input_shape[0]

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
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                        keras.models.save_model(self, fd.name, overwrite=True)
                        model_str = fd.read()
                d = {'model_str': model_str}
                return d

        def __setstate__(self, state):
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                        fd.write(state['model_str'])
                        fd.flush()
                        model = keras.models.load_model(fd.name, custom_objects = {'tf': tf, 'GCN_Layer': GCN_Layer})
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
                "description": "graph convolutional network",
                # ask about naming convention
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
                "hyperparams_to_tune": ["dimension", "beta", "alpha"]
        })
        
        def __init__(self, *, hyperparams : GCN_Hyperparams) -> None:
                super(GCN, self).__init__(hyperparams = hyperparams)

        
        def set_training_data(self, *, inputs : Input, outputs : Output) -> None:
                #self._adj = None
                #self._input = None

                learning_df, nodes_df, edges_df = self._parse_inputs(inputs)
                

                nodes_df = nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]
                node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]

                self._num_training_nodes = node_subset.values.shape[0]
                self._adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0], node_subset = node_subset.values.astype(np.int32))
                self._input = self._make_input_features(nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)])#.index])#, node_subset = node_subset)

                # dealing with outputs
                #if self._task in ['clf', 'class', 'classification', 'node_clf']:

                target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                use_outputs = False
                if use_outputs:
                        targets =  get_columns_of_type(outputs, target_types)
                else:
                        targets =  get_columns_of_type(learning_df, target_types)
                        
                
                self._label_unique = np.unique(targets.values).shape[0]
                self.label_encode = LabelEncoder()
                self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                
                #self.training_outputs = to_categorical(self.label_encode.fit_transform(outputs), num_classes = np.unique(outputs.values).shape[0])
                #else:
                #        raise NotImplementedError()

                self.fitted = False

        def _make_adjacency(self, edges_df, num_nodes = None, tensor = False, #True, 
                            node_subset = None):
                

                source_types = ('https://metadata.datadrivendiscovery.org/types/EdgeSource',
                                                'https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource',
                                                'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource',
                                                'https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource',
                                                'https://metadata.datadrivendiscovery.org/types/MultiEdgeSource')

                #sources = edges_df['source']
                sources = get_columns_of_type(edges_df, source_types)
                
                dest_types = ('https://metadata.datadrivendiscovery.org/types/EdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget',
                                                'https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget')
                dests = get_columns_of_type(edges_df, dest_types)

                
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
                

                if tensor:
                        adj = tf.SparseTensor([[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])], [1.0 for i in range(sources.values.shape[0])], dense_shape = (num_nodes, num_nodes))
                else:
                        adj = csr_matrix(([1.0 for i in range(sources.values.shape[0])], ([sources.values[i, 0] for i in range(sources.values.shape[0])], [dests.values[i,0] for i in range(sources.values.shape[0])])), shape = (num_nodes, num_nodes), dtype = np.float32)
                #tf.sparse.placeholder(dtype,shape=None,name=None)
        
                return adj
                # PREVIOUS RETURN
                self._adj = keras.layers.Input(tensor = adj if tensor else tf.convert_to_tensor(adj), sparse = True)

                #if self._adj is None:                
                #else:
                        #self._adj = #tf.assign(self._adj, adj if tensor else tf.convert_to_tensor(adj))
                        

                #raise NotImplementedError
                #return keras.layers.Input()

        def _make_input_features(self, nodes_df, tensor = False):# tensor = True):
        
                if tensor:
                        node_id = tf.cast(tf.eye(nodes_df.shape[0]), dtype = tf.float32)
                        #node_id = tf.sparse.eye(nodes_df.shape[0])
                else:
                        #node_id = scipy.sparse.identity(nodes_df.shape[0], dtype = np.float32) #
                        node_id = np.eye(nodes_df.shape[0])

                # preprocess features, e.g. if non-numeric / text?
                if len(nodes_df.columns) > 2:
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
                #if self._input is None:
                self._input = keras.layers.Input(tensor = to_return if tensor else tf.convert_to_tensor(to_return))
                #else:
                #        tf.assign(self._input, to_return if tensor else tf.convert_to_tensor(to_return))


        def fit(self, *, timeout : float = None, iterations : int = None) -> None:

                make_keras_pickleable()
                
                if self.fitted:
                        return CallResult(None, True, 1)

                # ******************************************
                # Feel free to fill in less important hyperparameters or example architectures here
                # ******************************************
                self._task = 'classification'
                self._act = 'relu'
                self._epochs = 200 if self._num_training_nodes < 10000 else 50
                self._units = [100, 100]
                self._mix_hops = 3
                self._modes = 1
                self._lr = 0.001
                self._optimizer = keras.optimizers.Adam(self._lr)
                self._extra_fc = 100 
                # self._adj and self._input already set as keras Input tensors

                adj_input = keras.layers.Input(shape = self._adj.shape[1:], name = 'adjacency', sparse = True)
                feature_input = keras.layers.Input(shape = self._input.shape[1:], name = 'features')#sparse =True) #tensor =  if tensor else tf.convert_to_tensor(to_return))
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



                outputs = []
                loss_functions = []
                loss_weights = []

                outputs.append(y_pred)

                if label_act == 'softmax':  # if self._task == 'node_clf': 
                        loss_functions.append(keras.objectives.categorical_crossentropy)
                else: #f label_act == 'sigmoid':
                        loss_functions.append(keras.objectives.binary_crossentropy)
                #    loss_functions.append(keras.objectives.mean_squared_error)#mse                                            
                
                loss_weights.append(1.0)

                # fit keras
                self.model = keras.models.Model(inputs = [adj_input, feature_input], outputs = outputs)
                self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)

                self.model.fit([self._adj, self._input], # already specified as tensors
                               [self.training_outputs]*len(outputs),                               
                               shuffle = False, epochs = self._epochs, batch_size = self._num_training_nodes) # validation_data = [

                self.fitted = True
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
                if self.fitted:
                        # embed ALL (even unlabelled examples)
                        learning_df, nodes_df, edges_df = self._parse_inputs(inputs)
                        node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]
                                        
                        adj = self._make_adjacency(edges_df, num_nodes = nodes_df.shape[0], node_subset = node_subset.values.astype(np.int32))
                        inp = self._make_input_features(nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]) #.index])
                        
                        #result = self._embedding_model.predict()
                        
                        try:
                                output_pred = self.model.layers[-1].output # all layer outputs
                                func = K.function([self.model.input[0], self.model.input[1], K.learning_phase()], [output_pred])
                                result = func([adj, inp, 1.])[0]
                        except: # could be preferable, but both prediction methods should work
                                result = self.model.predict([adj, inp])
                        
                        result = np.argmax(result, axis = -1) #if not self.hyperparams['return_embedding'] else result
                        
                        if self.hyperparams['return_embedding']:
                                output_embed = self.model.layers[-2].output 
                                func = K.function([self.model.input[0], self.model.input[1], K.learning_phase()], [output_embed])
                                embed = func([adj, inp, 1.])[0]
                                print("RESULT SHAPE ", result.shape)
                                print("EMBED SHAPE ", embed.shape)
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
                print("Output SHAPE ", output.shape)
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
                        weights = self.model.get_weights(),
                        adj = self._adj)
        
        def set_params(self, *, params: GCN_Params) -> None:

                # assign model attributes (e.g. in loading from pickle)

                self.fitted = params['fitted']
                self.model = params['model']
                self.model.set_weights(params['weights'])
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


         
