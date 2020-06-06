import os
import sys
import typing
import numpy as np
import pdb

import tensorflow as tf
#import tensorflow as tf#
import tensorflow.keras as keras #compat.v1. 
import pandas as pd
import copy 
import importlib

import keras.objectives
import keras.backend as K
from sklearn import preprocessing
import tempfile
import scipy.sparse
from scipy.sparse import csr_matrix
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import keras.models

from common_primitives import utils
import d3m.container as container
from d3m.metadata.base import CONTAINER_SCHEMA_VERSION, DataMetadata, ALL_ELEMENTS, SelectorSegment
from d3m.base import utils as base_utils
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import List as d3m_List
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration, LogUniform
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
#import IPython

import dsbox_graphs.config_ as cfg_

#tf.logging.set_verbosity(tf.logging.ERROR)

#tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Input = container.Dataset 
Output = container.DataFrame


def dot(x, y, sparse=False):
        """Wrapper for tf.matmul (sparse vs dense)."""
        if sparse:
                try:
                        res = tf.sparse.sparse_dense_matmul(x, y) 
                except Exception as e:
                        try:
                                res = tf.matmul(x, y, a_is_sparse = True)
                        except:
                                res = tf.matmul(x, y)
                                #x = tf.contrib.layers.dense_to_sparse(x)
                                #res = tf.sparse_tensor_dense_matmul(x, y)
        else:
                        res = tf.matmul(x, y) #K.dot(x,y) 
        return res

def sparse_exponentiate(inputs, exponent = 1, sparse = False):
        adj = inputs[0]
        x = inputs[1]
        res = x
        if exponent == 0:
                        return res

        for k in range(exponent):
                        res = dot(adj, res, sparse = sparse)
        return res

def identity(x_true, x_pred):
                return x_pred

# selects only those 
def semi_supervised_slice(inputs, first = None):
        # input as [tensor, indices_to_select]
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
        return tf.cast(tf.reshape(sliced, [-1, tf.shape(input=tensor)[-1]]), tf.float32)

def assign_scattered(inputs):
        # "Undo" slice.  Used on loss function to give calculated loss for supervised examples, else 0 
        # inputs = [loss_on_slices, shape_ref, indices]
        slice_loss = inputs[0]
        shape_ref = inputs[1]
        # e.g. loss goes in batch dim 0,2,4,6,8, inds.shape = (5,1)
        inds = tf.expand_dims(tf.cast(inputs[-1], tf.int32), -1)
        full_loss = tf.scatter_nd(inds, 
                                        slice_loss, 
                                        shape = [tf.shape(input=shape_ref)[0]])
        return full_loss #tf.reshape(full_loss, (-1,))


def import_loss(inputs, function = None, first = None):
        if isinstance(function, str):
                        import importlib
                        mod = importlib.import_module('keras.objectives')
                        function = getattr(mod, function)
        try:
                        return function(inputs[0], inputs[-1]) if function is not None else inputs
        except:
                        inputs[0] = tf.gather(inputs[0], np.arange(first))
                        return function(inputs[0], inputs[-1]) if function is not None else inputs




def _update_metadata(metadata: DataMetadata, resource_id: SelectorSegment) -> DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
                raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                        type=resource_metadata.get('structural_type', None),
                ))

        resource_metadata.update(
                {
                        'schema': CONTAINER_SCHEMA_VERSION,
                },
        )

        new_metadata = DataMetadata(resource_metadata)

        new_metadata = metadata.copy_to(new_metadata, (resource_id,))

        # Resource is not anymore an entry point.
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata

def get_resource(inputs, resource_name):
        _id, _df = base_utils.get_tabular_resource(inputs, resource_name)
        _df.metadata = _update_metadata(inputs.metadata, _id)
        return _id, _df

def get_columns_not_of_type(df, semantic_types): 
        # NOTE: Fails quietly in case of no metadata (doesn't remove columns)

        columns = df.metadata.list_columns_with_semantic_types(semantic_types)

        def can_use_column(column_index: int) -> bool:
                return column_index not in columns

        # hyperparams['use_columns'], hyperparams['exclude_columns']
        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(df.metadata, [], [], can_use_column) 

        if not columns_to_use:
                        raise ValueError("Input data has no columns matching semantic types: {semantic_types}".format(
                                        semantic_types=semantic_types,
                        ))


        return df.select_columns(columns_to_use)


def get_columns_of_type(df, semantic_types): 
        columns = df.metadata.list_columns_with_semantic_types(semantic_types)

        def can_use_column(column_index: int) -> bool:
                return column_index in columns

        # hyperparams['use_columns'], hyperparams['exclude_columns']
        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(df.metadata, [], [], can_use_column) 

        if not columns_to_use:
                        raise ValueError("Input data has no columns matching semantic types: {semantic_types}".format(
                                        semantic_types=semantic_types,
                        ))


        return df.select_columns(columns_to_use)

def make_keras_pickleable():
                def __getstate__(self):
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
                                                                {'tf': tf, 'identity': identity, #'GCN_Layer': GCN_Layer,
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
                
                # hidden_layers = List( UniformInt )
                
                # epochs
                epochs = UniformInt(
                                lower = 10,
                                upper = 500,
                                default = 100,
                                #q = 5e-8,                                                                                                                                                                 
                                description = 'number of epochs to train',
                                semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )
                
                lr = LogUniform(
                        lower = 0.0001,
                        upper = 0.02,
                        default = 0.001,
                        description='learning rate for Adam optimization',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                )

                batch_norm = UniformBool(
                                default = True,
                                description='use batch normalization',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                )
                
                use_features = UniformBool(
                                default = True,
                                description='Indicates whether to use input features.  If False, uses only adjacency matrix.  This is also a workaround for not having features from all input points (see data-supply/issues/213)',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                )

                include_adjacency = UniformBool(
                                default = False,
                                description='Indicates whether to use adjacency matrix as part of feature input X. Will behave as if True if use_features=False or only a subset of nodes features are available.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                )
                

                return_embedding = UniformBool(
                                default = True,
                                description='return embedding alongside classification prediction.  Both may be treated as input features to downstream classifier.  If False, this primitive is used as a classifier directly',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                )
                
                line_graph = UniformBool(
                                default = False,
                                description='treat edges as nodes, construct adjacency matrix based on shared edges.  relevant for edge based classification, e.g. link prediction.  NOTE: Primitive does not work out of the box for multi-edge link prediction.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                )
                # node_subset = UniformBool(
                #               default = True,
                #               description=' treat only labeled examples (which somewhat defeats purpose of graph convolution, but is a workaround for incomplete feature data : https://datadrivendiscovery.slack.com/archives/C4QUVR65N/p1572991617079000',
                #               semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                # )
                sparse = UniformBool(
                                default = False,
                                description='try using sparse adjacency matrix',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
                )


class GCN(SupervisedLearnerPrimitiveBase[Input, Output, GCN_Params, GCN_Hyperparams]):
                """
                See base classes here : 
                                https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/primitive_interfaces

                """

                metadata = PrimitiveMetadata({
                        "schema": "v0",
                        "id": "48572851-b86b-4fda-961d-f3f466adb58e",
                        "version": "1.0.0",
                        "name": "Mixhop GCN",
                        "description": "Graph convolutional neural networks (GCN) as in Kipf & Welling 2016, generalized to k-hop edge links via Abu-el-Haija et al 2019: https://arxiv.org/abs/1905.00067 (GCN recovered for k = 1).  In particular, learns weight transformation of feature matrix X for various powers of adjacency matrix, i.e. nonlinearity(A^k X W), and concatenates into an embedding layer.  Feature input X may be of the form: identity matrix (node_id) w/ node features appended as columns.  Specify order using 'adjacency_order' hyperparam.  Expects list of [learning_df, edges_df, edges_df] as input (e.g. by running common_primitives.normalize_graphs + data_tranformation.graph_to_edge_list.DSBOX)",
                        "python_path": "d3m.primitives.feature_construction.gcn_mixhop.DSBOX",
                        "original_python_path": "gcn_mixhop.GCN",
                        "can_use_gpus": True,
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
                        learning_df, nodes_df, edges_df = self._parse_inputs(inputs)

                        #nodes_df = nodes_df.loc[learning_df['d3mIndex'].astype(np.int32)]

                        ''' *******************************************
                                                        NODE SUBSET
                                - shouldn't be necessary if restricting to adj matrix?
                                - check if have all features? (e.g. learning_df.shape[0] == sources+dests.unique)
                                *******************************************
                        '''
                        node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]

                        #try:
                        target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')

                        features_df = get_columns_not_of_type(learning_df, target_types)
                        features_df = features_df.iloc[:, 2:] if 'nodeID' in features_df.columns and 'd3mIndex' in features_df.columns else features_df


                
                        self.node_encode = LabelEncoder()

                        sources, dests = self._get_source_dest(edges_df)
                        sources = sources.astype(np.int32)
                        dests = dests.astype(np.int32)
                        to_fit = np.sort(np.concatenate([sources.values,dests.values], axis = -1).astype(np.int32).ravel())
 

                        #( Hacky workaround for edges_df / learning_df ID mismatch )
                        if np.amin(to_fit) == 1 and 'nodeID' in learning_df.columns and int(np.amin(learning_df['nodeID'].values)) == 0:
                                edges_df['node1'] = edges_df['node1'].values.astype(int) - 1
                                edges_df['node2'] = edges_df['node2'].values.astype(int) - 1
                                sources['node1']-=1
                                dests['node2']-=1
                                to_fit -= 1


                        self.node_encode.fit(to_fit) 

                        node_subset_enc = self.node_encode.transform(node_subset.values.astype(np.int32))
                        sources[sources.columns[0]] = self.node_encode.transform(sources.values.astype(np.int32))
                        dests[dests.columns[0]] = self.node_encode.transform(dests.values.astype(np.int32))
                        self._num_edge_nodes = len(list(self.node_encode.classes_))

                        
                        if self.hyperparams['line_graph']:
                                self._num_training_nodes = edges_df.values.shape[0]
                                self._adj = self._make_line_adj(edges_df) 
                                self._input = self._make_line_inputs(edges_df)
                        else:
                                # ****** TO DO ******** : node subset?
                                self._num_training_nodes = nodes_df.values.shape[0]     
                                
                                self.full_adj = self._make_adjacency(sources,dests)  
                                
                                # features_df removes index, labels
                                self._input = self._make_input_features(features_df,
                                                                        just_adj = not self.hyperparams['use_features'],
                                                                        incl_adj = self.hyperparams['include_adjacency'])


                                self._adj = self.full_adj[np.ix_(node_subset_enc, node_subset_enc)]

                                # renormalize after taking node subsets

                                self._adj = self._normalize_adjacency(self._adj)


                        target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                        targets = get_columns_of_type(learning_df, target_types)
        
                        self._parse_data(learning_df, targets, node_subset = node_subset)
                        self.fitted = False

                def _parse_data(self, learning_df, targets, node_subset = None):


                        self.training_inds = np.arange(learning_df.shape[0])

                        self._label_unique = np.unique(targets.values).shape[0]
                        #  ******  LABEL ENCODE ?  ****** 
                        
                        try:
                                        self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                        except:
                                        self.label_encode = LabelEncoder()
                                        self.training_outputs = to_categorical(self.label_encode.fit_transform(targets.values), num_classes = np.unique(targets.values).shape[0])
                                

                        self._num_labeled_nodes = self.training_outputs.shape[0]


                        # CREATE INPUT TENSORS FOR KERAS TRAINING
                        self.outputs_tensor = tf.constant(self.training_outputs)
                        self.inds_tensor = tf.constant(np.squeeze(self.training_inds), dtype = tf.int32)
                        
                        #self.y_true = keras.layers.Input(tensor = self.outputs_tensor, name = 'y_true', dtype = 'float32')
                        #self.inds = keras.layers.Input(tensor = self.inds_tensor, dtype='int32', name = 'training_inds')


                def _parse_inputs(self, inputs : Input):
                        # Input is a dataset now
                        try:
                                learning_id, learning_df = get_resource(inputs, 'learningData')
                        except:
                                pass
                        try: # resource id, resource
                                nodes_id, nodes_df = get_resource(inputs, '0_nodes')
                        except:
                                try:
                                        nodes_id, nodes_df = get_resource(inputs, 'nodes')
                                except:
                                        nodes_df = learning_df
                        try:
                                edges_id, edges_df = get_resource(inputs, '0_edges')
                        except:
                                try:
                                        edges_id, edges_df = get_resource(inputs, 'edges')
                                except:
                                        edges_id, edges_df = get_resource(inputs, '1')

                        return learning_df, nodes_df, edges_df

                def _get_source_dest(self, edges_df, source_types = None, dest_types = None):
                                
                                if source_types is None:
                                                source_types = ('https://metadata.datadrivendiscovery.org/types/EdgeSource',
                                                                                'https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource',
                                                                                'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource',
                                                                                'https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource',
                                                                                'https://metadata.datadrivendiscovery.org/types/MultiEdgeSource')

                        
                                sources = get_columns_of_type(edges_df, source_types)
                        
                                if dest_types is None:
                                                dest_types = ('https://metadata.datadrivendiscovery.org/types/EdgeTarget',
                                                                                                'https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget',
                                                                                                'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget',
                                                                                                'https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget',
                                                                                                'https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget')
                                dests = get_columns_of_type(edges_df, dest_types)
                                
                                return sources, dests


                def _normalize_adjacency(self, adj = None, node_subset = None):
                        if adj is None:
                                adj = self._adj

                        if isinstance(adj, np.ndarray):
                                row_sum = np.sqrt(np.sum(adj,axis=-1).ravel())
                                col_sum = np.sqrt(np.sum(adj,axis=0).ravel())
                                rows = np.diag(np.where(np.isinf(1/row_sum), np.zeros_like(row_sum), 1/row_sum))
                                cols = np.diag(np.where(np.isinf(1/col_sum), np.zeros_like(col_sum), 1/col_sum))
                                adj = np.dot(np.dot(rows, adj), cols)
                        
                        else:
                                row_sum = np.sqrt(adj.sum(axis=-1).A.ravel())
                                col_sum = np.sqrt(adj.sum(axis=0).A.ravel())
                                rows = scipy.sparse.diags(np.where(np.isinf(1/row_sum), np.zeros_like(row_sum), 1/row_sum)) 
                                cols = scipy.sparse.diags(np.where(np.isinf(1/col_sum), np.zeros_like(col_sum), 1/col_sum))
                                #degrees = scipy.sparse.diags(1/np.sqrt(adj.sum(axis=-1).A.ravel())).multiply(scipy.sparse.diags(1/np.sqrt(adj.sum(axis=0).A.ravel())))
                                #adj = adj @ degrees
                                adj = rows.dot(adj).dot(cols)
                        return adj

                def _make_adjacency(self, sources, dests, num_nodes = None, tensor = False, #True, 
                                                        node_subset = None):
                                

                        sources = sources.astype(np.int32)
                        dests = dests.astype(np.int32)
                        
                        num_nodes = np.unique(np.concatenate([sources, dests], axis = -1)).shape[0] if num_nodes is None else num_nodes
                        
                        
                        if tensor:
                                adj = tf.SparseTensor([[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])], [1.0 for i in range(sources.values.shape[0])], dense_shape = (num_nodes, num_nodes))
                        else:
                                self_connect = [i for i in np.sort(np.unique(sources.values.astype(np.int32)))] if len(np.unique(sources.values.astype(np.int32)))>len(np.unique(dests.values.astype(np.int32))) else [i for i in np.sort(np.unique(dests.values.astype(np.int32)))] 
                                source_inds = [sources.values.astype(np.int32)[i, 0] for i in range(sources.values.shape[0])]
                                dest_inds = [dests.values.astype(np.int32)[i,0] for i in range(sources.values.shape[0])]

                                
                                # ************** TREATS ALL EDGES AS SYMMETRIC, UNWEIGHTED **********************************
                                # to do : fix
                                # adds self-connections
                                entries = np.concatenate([np.array([source_inds, dest_inds]), np.array([self_connect, self_connect])],axis = -1)
                                
                                if self.hyperparams['sparse']:
                                        adj = csr_matrix(([1.0 for i in range(entries.shape[-1])], #range(source.values.shape[0])], 
                                                          entries), shape = (num_nodes, num_nodes), dtype = np.float32)
                                else:
                                        adj = np.zeros(shape = (num_nodes,num_nodes))
                                        for i in range(entries.shape[-1]):
                                                adj[entries[0,i], entries[1,i]]=1.0
                                                adj[entries[1,i], entries[0,i]]=1.0 # remove?
                                                                                

                        return adj
                
                def _make_line_adj(self, edges_df, node_subset = None, tensor = False):
                        sources, dests = self._get_source_dest(edges_df)
                        
                        num_nodes = edges_df.shape[0]

                        # TO DO: change edge detection logic to reflect directed / undirected edge source/target
                        #   multigraph = different adjacency matrix (and weights) for each edge type   (e.g. link prediction)
                        
                        edges = [[i,j] for i in range(sources.values.shape[0]) for j in range(dests.values.shape[0]) if dests.values[j,0] == sources.values[i,0]]
                        weights = [1.0 for i in range(len(edges))]
                        
                        if tensor:
                                adj = tf.SparseTensor(edges, weights, dense_shape = (num_nodes, num_nodes))
                        else:
                                edges = ([edges[i][0] for i in range(len(edges))], [edges[i][1] for i in range(len(edges))])
                                adj = csr_matrix((weights, edges), shape = (num_nodes, num_nodes), dtype = np.float32)
                        return adj


                def _make_input_features(self, nodes_df, tensor = False, num_nodes = None, just_adj = False, incl_adj = False):# tensor = True):
                        num_nodes = num_nodes if num_nodes is not None else nodes_df.shape[0]

                        # include adjacency matrix as features?  Or default to this if don't have 
                        if incl_adj or (len(nodes_df.columns) > 2 and not (self._num_edge_nodes == self._num_training_nodes and just_adj)):
                                if tensor:
                                        node_id = tf.cast(tf.eye(num_nodes), dtype = tf.float32)
                                        #node_id = tf.sparse.eye(nodes_df.shape[0])
                                else:
                                        #node_id = scipy.sparse.identity(nodes_df.shape[0], dtype = np.float32) #
                                        node_id = np.eye(num_nodes)

                        
                                self._input_columns = num_nodes #if incl_adj else 0
                        # TO DO: preprocess features, e.g. if non-numeric / text?
                        # CHANGE (included but to be tested): inputs = adjacency if don't have features for all nodes
                        if len(nodes_df.columns) > 2 and (self._num_edge_nodes == self._num_training_nodes and not just_adj):

                                #try: # take semantic types = Attribute if possible
                                semantic_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                                                  'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')

                                features = get_columns_of_type(nodes_df, semantic_types).values.astype(np.float32)
                                #except: # Goal is just to remove label

                                #        features = nodes_df.remove_columns(['label'])

                                self._input_columns = self._input_columns+features.shape[-1] if incl_adj else features.shape[-1]

                                if tensor:
                                                features = tf.convert_to_tensor(value=features)
                                                to_return= tf.concat([features, node_id], -1) if incl_adj else features
                                else:
                                                to_return= np.concatenate([features, node_id], axis = -1) if incl_adj else features
                        else:
                                to_return = node_id

                                        
                        return to_return


                # line graph switches roles of edges and nodes (e.g. for link prediction)
                # ******* NOT TESTED ************
                def _make_line_inputs(self, edges_df, tensor = False, incl_adj = True):
                        # ID for evaluating adjacency matrix
                        if tensor:
                                        node_id = tf.cast(tf.eye(edges_df.shape[0]), dtype = tf.float32)
                        else:
                                        node_id = np.eye(edges_df.shape[0])
                        
                        # additional features?
                        # preprocess features, e.g. if non-numeric / text?
                        if len(edges_df.columns) > 2:
                                        semantic_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                                                                          'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')

                                        features = get_columns_of_type(edges_df, semantic_types).values.astype(np.float32)
                                        
                                        if tensor:
                                                        features = tf.convert_to_tensor(value=features)
                                                        to_return= tf.concat([features, node_id], -1)
                                        else:
                                                        to_return=np.concatenate([features, node_id], axis = -1)
                        else:
                                        to_return = node_id


                        return to_return



                def fit(self, *, timeout : float = None, iterations : int = None) -> None:

                        if self.fitted:
                                return CallResult(None, True, 1)

                        self._task = 'classification'
                        self._act = 'tanh'
                        self._epochs = self.hyperparams['epochs']
                        self._units = [100, 100, 100]
                        self._mix_hops = self.hyperparams['adjacency_order']
                        self._modes = 1
                        self._lr = self.hyperparams['lr']
                        self._optimizer = keras.optimizers.Adam(self._lr)
                        self._extra_fc = None #100
                        self._batch_norm = self.hyperparams['batch_norm']


                        # DEVEL option 
                        inp_tensors = False
                        if inp_tensors:
                                        adj_input = keras.layers.Input(tensor = self._adj, name = 'adjacency', dtype = 'float32')
                                        feature_input = keras.layers.Input(tensor = self._input, name = 'features', dtype = 'float32')

                                        #adj_input = keras.layers.Input(tensor = self._adj, batch_shape = (None, tf.shape(self._adj)[-1]), name = 'adjacency', dtype = tf.float32)
                                        #feature_input = keras.layers.Input(tensor = self._input, batch_shape = (None, self._input_columns), name = 'features', dtype = tf.float32)
                        else:
                                        adj_input = keras.layers.Input(shape = (self._num_training_nodes,), name = 'adjacency', dtype = 'float32', sparse = self.hyperparams['sparse'])
                                        feature_input = keras.layers.Input(shape = (self._input_columns,), name = 'features', dtype = 'float32')#, sparse =True)
                        
                        self.y_true = keras.layers.Input(tensor = self.outputs_tensor, name = 'y_true', dtype = 'float32')
                        self.inds = keras.layers.Input(tensor = self.inds_tensor, dtype='int32', name = 'training_inds')

                   
                        # **** TO DO **** utilize self._modes (e.g. for link prediction with multiple types)
                        A = adj_input
                        H = feature_input

                        for h_i in range(len(self._units)):
                                act_k = []
                                
                                for k in range(self._mix_hops+1):
                                        # try to accommodate different sizes per adjacency power
                                        if isinstance(self._units[h_i], list) or isinstance(self._units[h_i], np.ndarray):
                                                h_i_k = self._units[h_i][k]
                                        else:
                                                h_i_k = self._units[h_i]

                                        #pre_w = GCN_Layer(k=k)([A, H])
                                        pre_w = keras.layers.Lambda(sparse_exponentiate, name ='pre_w_exp_'+str(k)+'_'+str(h_i), arguments = {'exponent': k, 'sparse': self.hyperparams['sparse']})([A,H])

                                        # CHANGE feeding of _units
                                        act = keras.layers.Dense(h_i_k, activation = self._act, name='w'+str(k)+'_'+str(h_i))(pre_w)

                                        act_k.append(act)
                                H = keras.layers.Concatenate(axis = -1, name = 'mix_'+str(self._mix_hops)+'hops_'+str(h_i))(act_k)


                        if self._extra_fc is not None and self._extra_fc:
                                H = keras.layers.Dense(self._extra_fc, activation = self._act)(H)




                        # ********************** ONLY NODE CLF RIGHT NOW *******************************
 
                        label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
                        y_pred = keras.layers.Dense(self._label_unique, activation = label_act, name = 'y_pred')(H)

                        if self._task in ['classification', 'clf']:
                                if label_act == 'softmax':  # if self._task == 'node_clf': 
                                        loss_function = keras.objectives.categorical_crossentropy
                                else: 
                                        loss_function = keras.objectives.binary_crossentropy
                        else:
                                loss_function = keras.objectives.mean_squared_error

                        
                        # Note: Y-true is an input tensor
                        y_pred_slice = keras.layers.Lambda(semi_supervised_slice)([y_pred, self.inds])#, arguments = {'inds': self.training_inds})(y_pred)
                        
                        y_true_slice = keras.layers.Lambda(semi_supervised_slice)([self.y_true, self.inds])

                        slice_loss = keras.layers.Lambda(import_loss, arguments = {'function': loss_function, 'first': self._num_labeled_nodes})([y_true_slice, y_pred_slice])
                        
                        full_loss = keras.layers.Lambda(assign_scattered)([slice_loss, y_pred, self.inds])

                        outputs = []
                        loss_functions = []
                        loss_weights = []
                        outputs.append(full_loss)
                        loss_functions.append(identity)
                        loss_weights.append(1.0)
        
                         
                        # fit keras
                        self.model = keras.models.Model(inputs = [adj_input, feature_input, self.y_true, self.inds], 
                                                                                        outputs = outputs)      
                        self.pred_model = keras.models.Model(inputs =  [adj_input, feature_input, self.inds], 
                                                                                                 outputs = [y_pred_slice])    
                        self.embedding_model = keras.models.Model(inputs = [adj_input, feature_input], 
                                                                                                          outputs = [H])      
                        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)


                        try:
                                self._adj = self._adj.todense() if (not self.hyperparams['sparse'] and not isinstance(_adj,np.ndarray)) else self._adj
                        except Exception as e:
                                pass

                        
                        self.model.fit(x = [self._adj, self._input], 
                                                   y = [self.training_outputs],
                                                   shuffle = False, epochs = self._epochs, 
                                                   batch_size = self._num_training_nodes,
                                                   verbose = 1
                                ) 
                        
                        
                        self.fitted = True
                        make_keras_pickleable()
                        return CallResult(None, True, 1)

                
                def produce(self, *, inputs : Input, outputs : Output, timeout : float = None, iterations : int = None) -> CallResult[Output]:
                        make_keras_pickleable()
                        if self.fitted:
                                # embed ALL (even unlabelled examples)
                                learning_df, nodes_df, edges_df = self._parse_inputs(inputs)
                                
                                node_subset = learning_df[[c for c in learning_df.columns if 'node' in c and 'id' in c.lower()][0]]

                                target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                                features_df = get_columns_not_of_type(learning_df, target_types)
                                features_df = features_df.iloc[:, 2:] if 'nodeID' in features_df.columns and 'd3mIndex' in features_df.columns else features_df
                                #features_df = learning_df.remove_columns([learning_df.columns.get_loc(c) for c in learning_df.columns if 'node' in c and 'id' in c.lower() or 'd3mIndex' in c])
                                

                                
                                if not self.hyperparams['line_graph']:
                                        
                                        #self._num_training_nodes = node_subset.values.shape[0]
                                                
                                        _input_ = self._make_input_features(features_df, just_adj = not self.hyperparams['use_features'], incl_adj = self.hyperparams['include_adjacency'])
                                        #_input_ = self._make_input_features(nodes_df, just_adj = not self.hyperparams['use_features'], incl_adj = self.hyperparams['include_adjacency'])
                                        # PRODUCE CAN WORK ON ONLY SUBSAMPLED Adjacency matrix (already created)
                                        _nodes = self.pred_model.input_shape[0][-1]
                                        _features = self.pred_model.input_shape[1][-1]
                                        _input = np.zeros((_nodes,_features))
                                        
                                        node_subset_enc = self.node_encode.transform(node_subset.values.astype(np.int32))
                                        _input[node_subset_enc] = _input_
                                        _adj = self.full_adj
                                        
                                else:
                                        # LINE GRAPH WIP
                                        self._num_training_nodes = edges_df.values.shape[0]
                                        _adj = self._make_line_adj(edges_df) 
                                        _input = self._make_line_inputs(edges_df)
                                        raise NotImplementedError()     



                                target_types = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                                
                                targets =  get_columns_of_type(learning_df, target_types)
                                
                                        
                                self._parse_data(learning_df, targets, node_subset = node_subset)
                                _adj = _adj.todense() if (not self.hyperparams['sparse'] and not isinstance(_adj,np.ndarray)) else _adj
                                result = self.pred_model.predict([_adj, _input], steps = 1)#, batch_size = len(self.training_inds.shape[0]))
                                        
                                        
                                result = np.argmax(result, axis = -1) #if not self.hyperparams['return_embedding'] else result
                                   
                                if self.hyperparams['return_embedding']:
                                        try:
                                                embed = self.embedding_model.predict([_adj, _input], steps = 1)
                                        except:
                                                embed = self.embedding_model.predict([_adj, _input, self.training_inds], steps = 1)
                                        try:
                                                result = np.concatenate([result, embed], axis = -1)
                                        except:
                                                result = np.concatenate([np.expand_dims(result, 1), embed], axis = -1)
                                        #result = result[self.training_inds]

                        else:
                                        raise Error("Please call fit first")
                        
                        # ******************************************
                        # Subroutine to get output in proper D3M format

                        # ** Please confirm / double check **
                        # ******************************************

                        
                        target_types = ('https://metadata.datadrivendiscovery.org/types/TrueTarget',
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

                        learn_df = d3m_DataFrame(learning_df, generate_metadata = True)
                        learn_df = get_columns_not_of_type(learn_df, target_types)
                        
                        
                        result_df = d3m_DataFrame(result, generate_metadata = True)
                        result_df = result_df.loc[result_df.index.isin(learning_df['d3mIndex'].values)] 
                        

                        for column_index in range(result_df.shape[1]):
                                col_dict = dict(result_df.metadata.query((ALL_ELEMENTS, column_index)))
                                col_dict['structural_type'] = type(1.0)
                                col_dict['name'] = str(learn_df.shape[1] + column_index) #should just be column index, no corex prefix #'corex_' + 
                                col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')

                                result_df.metadata = result_df.metadata.update((ALL_ELEMENTS, column_index), col_dict)


                        #if len(result_df.index) != len(learn_df.index):
                        #        try:
                        #                learn_df = learn_df.get_loc(self.training_inds)
                        #        except Exception as e:
                        #                print(e)
                        #                print("learn_df = learn_df.get_loc(self.training_inds)")
                                        

                        result_df.index = learn_df.index.copy()
                        output = utils.append_columns(learn_df, result_df)
                        
                        return CallResult(output, True, 1)

                        # PREVIOUS RETURN MECHANISM
                        # outputs = output
                        
                        # self._training_indices = [c for c in learning_df.columns if isinstance(c, str) and 'index' in c.lower()]

                        # output = utils.combine_columns(return_result='new', #self.hyperparams['return_result'],
                        #         add_index_columns=True,#self.hyperparams['add_index_columns'], 
                        #         inputs=learning_df, columns_list=[output], source=self, column_indices=self._training_indices)
                   
                        # return CallResult(output, True, 1)

        
        
                                
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









