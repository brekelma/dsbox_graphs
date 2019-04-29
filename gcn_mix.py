import os
import sys
import typing
import networkx

import tensorflow as tf
import keras
import keras.backend as K
from sklearn import preprocessing
import tempfile

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
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

import config as cfg_



Input = container.List #container.DataFrame
Output = container.DataFrame
                 
                         
class GCN_Params(params.Params):

    ''' 
    Attributes necessary to resume training or run on test data (if loaded from pickle)

    Code specifications of parameters: 
        https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/params.py
    '''

    fitted: typing.Union[bool, None] # fitted required, set once primitive is trained
    model: typing.Union[keras.models.Model, None]

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
    return_list = UniformBool(
        default = False,
        description='for testing',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


# all primitives must be pickle-able, and this should do the trick for Keras models

def make_keras_pickleable():
    def __getstate__(self):
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
            model = keras.models.load_model(fd.name)#, custom_objects = {'tanh64': tanh64, 'log_sigmoid': tf.math.log_sigmoid, 'dim_sum': dim_sum, 'echo_loss': echo_loss, 'tf': tf, 'permute_neighbor_indices': permute_neighbor_indices})
        self.__dict__ = model.__dict__


    #cls = Sequential
    #cls.__getstate__ = __getstate__
    #cls.__setstate__ = __setstate__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__



def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def sparse_exp_ax(adj, x, exponent = 1):
    res = x
    for k in range(exponent):
        res = dot(adj, res, sparse = True)
    return res

def get_columns_of_type(df, semantic_types): 
    columns = df.metadata.list_columns_with_semantic_types(semantic_types)

    def can_use_column(column_index: int) -> bool:
        return column_index in columns

    # hyperparams['use_columns'], hyperparams['exclude_columns']
    columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, use_columns = [], exclude_columns=[], can_use_column)

    if not columns_to_use:
        raise ValueError("Input data has no columns matching semantic types: {semantic_types}".format(
            semantic_types=semantic_types,
        ))

    if columns_not_to_use: #and hyperparams['use_columns']
        cls.logger.warning("Node attributes skipping columns: %(columns)s", {
            'columns': columns_not_to_use,
        })

    return df.select_columns(columns_to_use)



class GCN(UnsupervisedLearnerPrimitiveBase[Input, Output, GCN_Params, GCN_Hyperparams]):
    """
    See base classes here : 
        https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/primitive_interfaces

    """

    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "",
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
        "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM", "LATENT_DIRICHLET_ALLOCATION"],
        "primitive_family": "FEATURE_CONSTRUCTION",
        "hyperparams_to_tune": ["dimension", "beta", "alpha"]
    })
    
    def __init__(self, *, hyperparams : GCN_Hyperparams) -> None:
        super(GCN, self).__init__(hyperparams = hyperparams)

	
    def set_training_data(self, *, inputs : Input) -> None:
        # Will change...
        if len(inputs) == 3:
            learning_df = inputs[0]
            nodes_df = inputs[1]
            edges_df = inputs[-2]
        elif len(inputs) == 2:
            nodes_df = inputs[1]
            edges_df = inputs[-2]
        else:
            print("********** GCN INPUTS ***********", inputs)
            raise ValueError("Check inputs to GCN")


        self._adj = self._make_adjacency(edges_df, nodes_df.shape[0])
        self._input = self._make_input_features(nodes_df)


        if self._task in ['clf', 'class', 'classification', 'node_clf']:
            self._label_unique = np.unique(outputs.values).shape[0]
            self.label_encode = preprocessing.LabelEncoder()
            self.training_outputs = to_categorical(self.label_encode.fit_transform(outputs), num_classes = np.unique(outputs.values).shape[0])
        else:
            raise NotImplementedError()

        self.fitted = False

    def _make_adjacency(self, edges_df, num_nodes):
        print()
        print("********************************")
        print(df.metadata)
        print("********************************")
        print()


        source_types = ('https://metadata.datadrivendiscovery.org/types/EdgeSource',
                        'https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource',
                        'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource',
                        'https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource',
                        'https://metadata.datadrivendiscovery.org/types/MultiEdgeSource')
        sources = get_columns_of_type(edges_df, source_types)

        dest_types = ('https://metadata.datadrivendiscovery.org/types/EdgeTarget',
                        'https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget',
                        'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget',
                        'https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget',
                        'https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget')
        dests = get_columns_of_type(edges_df, dest_types)


        attr_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                      'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')
        attrs = get_columns_of_type(edges_df, attr_types)


        adj = np.zeros((num_nodes, num_nodes))

        for row in sources.shape[0]:
            i = sources.values[row, :]
            j = dests.values[row,:]
            print(i,j)
            try:
                val = attrs.values[row,:]
                print("Weight ", val)
                success = True
            except Exception as e:
                print()
                print("Could not get edge weight / attribute : ", e)
                print()
                success = False

            adj[i,j] = 1 if not success else val

        self._adj_np = adj
        return keras.layers.Input(tensor = tf.convert_to_tensor(adj))
        #raise NotImplementedError
        #return keras.layers.Input()

    def _make_input_features(self, nodes_df):
        node_id = np.eye(nodes_df.shape[0])

        # preprocess features, e.g. if non-numeric / text?
        if len(node_id.columns) > 2:
            semantic_types = ('https://metadata.datadrivendiscovery.org/types/Attribute',
                              'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')

            features = get_columns_of_type(nodes_df, semantic_types).values

            return np.concatenate([features, node_id])
        else:
            return node_id


    def fit(self, *, timeout : float = None, iterations : int = None) -> None:
        
        make_keras_pickleable()
        
        if self.fitted:
            return CallResult(None, True, 1)

        # ******************************************
        # Feel free to fill in less important hyperparameters or example architectures here
        # ******************************************

        self._act = 'relu'
        self._batch = 100
        self._epochs = 100
        self._units = [100, 100]
        self._mix_hops = 3
        self._modes = 1
        self._lr = 0.0003
        self._optimizer = Adam(self._lr)

        # self._adj and self._input already set as keras Input tensors

        embedding = self._make_gcn(self, h_dims = self._units,
                        mix_hops = self._mix_hops, 
                        modes = self._modes)

        # make_task
        # ********************** ONLY NODE CLF RIGHT NOW *******************************
        #if self._task == 'node_clf':
    	label_act = 'softmax' if self._label_unique > 1 else 'sigmoid'
        y_pred = Dense(self._label_unique, activation = label_act, name = 'y_pred')(t)

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
        self.model = keras.models.Model(inputs = self._inp, outputs = outputs)
        self.model.compile(optimizer = self._optimizer, loss = loss_functions, loss_weights = loss_weights)

        # NOT READY TO FIT... 
        self.model.fit(self._model_input, # INPUTS # 
                        [self.training_outputs]*len(outputs),                               
                        shuffle = True, epochs = self._epochs, batch_size = self._batch) # validation_data = [

        self.fitted = True
        return CallResult(None, True, 1)
						 
						 
    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]:
        if self.fitted:
            result = self._model._Y
        else:
            raise Error("Please call fit first")
        
        # ******************************************
        # Subroutine to get output in proper D3M format

        # ** Please confirm / double check **
        # ******************************************


        if self.hyperparams['return_list']:
            result_np = container.ndarray(result, generate_metadata = True)
            return_list = d3m_List([result_np, inputs[1], inputs[2]], generate_metadata = True)        
            return CallResult(return_list, True, 1)
        else:
            result_df = d3m_DataFrame(result, generate_metadata = True)
            nodeIDs = inputs[1]
            result_df['nodeID'] = nodeIDs
            result_df = d3m_DataFrame(result_df, generate_metadata = True)
            
            return CallResult(result_df, True, 1)

        
        # TO DO : continue_fit, timeout
        
    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, timeout: float = None, iterations: int = None) -> MultiCallResult:
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, timeout : float = None, iterations : int = None) -> MultiCallResult:
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def get_params(self) -> GCN_Params:

        # fill in with model attributes 

        return GCN_Params(
            fitted = self.fitted,
            model = self._model)
	
    def set_params(self, *, params: GCN_Params) -> None:

        # assign model attributes (e.g. in loading from pickle)

        self.fitted = params['fitted']
        self._model = params['model']

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
    #     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    #     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    #     'labels_mask': tf.placeholder(tf.int32),
    #     'dropout': tf.placeholder_with_default(0., shape=()),
    #     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    #       } 


    def _make_gcn(self, h_dims = [100, 100], mix_hops = 5, modes = 1):
        a = self._adj
        x = self._input

        # where should number of modes be handled?


        for h_i in range(h_dims):
            act_k = []
            for k in range(mix_hops):
                pre_w = keras.layers.Lambda(_gcn_layer, arguments = {'k': k})(x)
                act = keras.layers.Dense(h_dims[h_i], activation = self._act, name='w'+str(k)+'_'+str(h_i))(pre_w)
                act_k.append(act)
            x = keras.layers.Concatenate(axis = -1, name = 'mix_exponents')(act_k)

        # embedding tensor (concatenation of k)
        return x


    def _gcn_layer(self, inputs, pre_w = None, exponent = 1):
        # keras wrapper for sparse mult
        x = inputs[0] if isinstance(inputs, list) else x
        # exponent for mixhop  
        return sparse_exp_ax(self._adj, x, exponent = k)        



