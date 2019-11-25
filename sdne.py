import os
import sys
import typing
import networkx
import numpy as np

import tensorflow as tf
#from GEM.gem.embedding import node2vec
from GEM.gem.embedding import sdne
#from GEM.gem.embedding import sdne_utils
import keras.models
import tempfile
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from d3m.base import utils as base_utils


from common_primitives import utils
import d3m.container as container
import d3m.metadata.base as mbase
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

from d3m.container import List as d3m_List
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform, UniformBool, UniformInt, Union, Enumeration
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

import _config as cfg_
#import config as cfg_

#CUDA_VISIBLE_DEVICES=""

Input = container.Dataset 
#Input = container.DataFrame
#Output = container.List #
Output = container.DataFrame
#container.List #DataFrame #typing.Union[container.DataFrame, None]


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
        
        return df.select_columns(columns_to_use)

def _update_metadata(metadata: mbase.DataMetadata, resource_id: mbase.SelectorSegment) -> mbase.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update(
            {
                'schema': mbase.CONTAINER_SCHEMA_VERSION,
            },
        )

        new_metadata = mbase.DataMetadata(resource_metadata)

        new_metadata = metadata.copy_to(new_metadata, (resource_id,))

        # Resource is not anymore an entry point.
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata

def get_resource(inputs, resource_name):
    _id, _df = base_utils.get_tabular_resource(inputs, resource_name)
    _df.metadata = _update_metadata(inputs.metadata, _id)
    return _id, _df


def loadGraphFromEdgeDF(df, directed=True):
    graphtype = networkx.DiGraph if directed else networkx.Graph

    G = networkx.from_pandas_edgelist(df, edge_attr=True)#, create_using= graph_type)
    return G					 
						 
class SDNE_Params(params.Params):
    fitted: typing.Union[bool, None]
    #model: typing.Union[keras.models.Model, None]
    model: typing.Union[sdne.SDNE, None]
    node_encode: typing.Union[LabelEncoder, None]
# SDNE takes embedding dimension (d), 
# seen edge reconstruction weight (beta), 
# first order proximity weight (alpha), 
# lasso regularization coefficient (nu1), 
# ridge regreesion coefficient (nu2), 
# number of hidden layers (K), 
# size of each layer (n_units), 
# number of iterations (n_ite), 
# learning rate (xeta), 
# size of batch (n_batch), 
# location of modelfile 
# and weightfile save (modelfile and weightfile) as inputs

class SDNE_Hyperparams(hyperparams.Hyperparams):
    dimension = UniformInt(
        lower = 10,
        upper = 200,
        default = 10,
        #q = 5,
        description = 'dimension of latent embedding',
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )
    epochs = UniformInt(
        lower = 1,
        upper = 500,
        default = 50,
        #q = 5e-8,                                                                                                                                                                                        
        description = 'number of epochs to train',
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    beta = UniformInt( 
        lower = 1,
        upper = 20,
        default = 5,
        #q = 1,
        description = 'seen edge reconstruction weight (to account for sparsity in links for reconstructing adjacency.  matrix B in Wang et al 2016',
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
    lr = Uniform(
        lower = 1e-5,
        upper = 1e-2,
        default = 5e-4,
        #q = 5e-8,                                                                                                                                                                                               
        description = 'learning rate (constant across training)',
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )
    depth = UniformInt(
        lower = 1,
        upper = 10,
        default = 3,
        #q = 5,
        description = 'number of hidden layers',
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/ControlParameter']
        )
    return_list = UniformBool(
        default = False,
        description='for testing',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

class SDNE(UnsupervisedLearnerPrimitiveBase[Input, Output, SDNE_Params, SDNE_Hyperparams]):
    """
    Graph embedding method
    """

    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "7d61e488-b5bb-4c79-bad6-f1dc07292bf4",
        "version": "1.0.0",
        "name": "SDNE",
        "description": "Structural Deep Network Embedding (Wang et al 2016): unsupervised network embedding using autoencoders to preserve first order proximity (i.e. connected nodes have similar embeddings) and second order proximity (i.e. nodes with similar neighbors have similar embeddings).  Hyperparam alpha controls weight of 1st order proximity loss (L2 norm of embedding difference), beta controls second-order loss (reconstruction of adjacency matrix row, matrix B in Wang et al).  Expects list of [learning_df, nodes_df, edges_df] as input (e.g. by running common_primitives.normalize_graphs + data_tranformation.graph_to_edge_list.DSBOX)",
        "python_path": "d3m.primitives.feature_construction.sdne.DSBOX",
        "original_python_path": "sdne.SDNE",
        "source": {
            "name": "ISI",
            "contact": "mailto:brekelma@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_graphs" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": ["AUTOENCODER"],
        "primitive_family": "FEATURE_CONSTRUCTION",
        "hyperparams_to_tune": ["dimension", "beta", "alpha"]
    })
    
    def __init__(self, *, hyperparams : SDNE_Hyperparams) -> None:
        super(SDNE, self).__init__(hyperparams = hyperparams)
        # nu1 = 1e-6, nu2=1e-6, K=3,n_units=[500, 300,], rho=0.3, n_iter=30, xeta=0.001,n_batch=500
	
    def _make_adjacency(self, sources, dests, num_nodes = None, tensor = True):
        if num_nodes is None:
            num_nodes = len(self.node_encode.classes_)
        if tensor:
            try:
                adj = tf.SparseTensor([[sources.values[i, 0], dests.values[i,0]] for i in range(sources.values.shape[0])], [1.0 for i in range(sources.values.shape[0])], dense_shape = (num_nodes, num_nodes)) 
            except:
                adj = tf.SparseTensor([[sources[i], dests[i]] for i in range(sources.shape[0])], [1.0 for i in range(sources.shape[0])], dense_shape = (num_nodes, num_nodes)) 
        else:
            try:
                adj = csr_matrix(([1.0 for i in range(sources.values.shape[0])], ([sources.values[i, 0] for i in range(sources.values.shape[0])], [dests.values[i,0] for i in range(sources.values.shape[0])])), shape = (num_nodes, num_nodes))
            except:
                adj = csr_matrix(([1.0 for i in range(sources.shape[0])], ([sources[i] for i in range(sources.shape[0])], [dests[i] for i in range(sources.shape[0])])), shape = (num_nodes, num_nodes))
        return adj
    
    def _get_source_dest(self, edges_df, source_types = None, dest_types = None):
        print()
        print(edges_df.metadata)#.list_columns_with_semantic_types())
        try:
            print(edges_df[ALL_ELEMENTS])
        except:
            pass
        print("METADATA ")
        print()
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

    def _parse_inputs(self, inputs : Input, return_all = False):
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

        #return learning_df, nodes_df, edges_df
            

        # try:
        #     G = loadGraphFromEdgeDF(edges_df)
        # except Exception as e:
        #     print()
        #     print("***************** LOADING GRAPH FROM EDGE LIST error ************", e)
        #     print()

        # #self.training_data = G
        
        self.node_encode = LabelEncoder()
        sources, dests = self._get_source_dest(edges_df)
        sources = sources.astype(np.int32)[:100]
        dests = dests.astype(np.int32)[:100]
        to_fit = np.sort(np.concatenate([sources.values,dests.values], axis = -1).astype(np.int32).ravel())
        
        self.node_encode.fit(to_fit) #nodes_df[id_col].values)
        
        sources[sources.columns[0]] = self.node_encode.transform(sources.values.astype(np.int32))
        dests[dests.columns[0]] = self.node_encode.transform(dests.values.astype(np.int32))
        #id_col = [i for i in nodes_df.columns if 'node' in i and 'id' in i.lower()][0]
        #self.node_enc.fit(nodes_df[id_col].values)

        other_training_data = self._make_adjacency(sources,dests, tensor = False)
        return other_training_data if not return_all else other_training_data, learning_df, nodes_df, edges_df

    def set_training_data(self, *, inputs : Input) -> None:

        training_data = self._parse_inputs(inputs)
        if isinstance(training_data, tuple):
            training_data = training_data[0]
        try:
            self.training_data = networkx.from_scipy_sparse_matrix(training_data)
        except Exception as e:
            print()
            print(e)
            print()
            import IPython
            IPython.embed()

        self.fitted = False

    def fit(self, *, timeout : float = None, iterations : int = None) -> None:
        
        if self.fitted:
            return CallResult(None, True, 1)

        args = {}
        args['nu1'] = 1e-6
        args['nu2'] = 1e-6
        args['K'] = self.hyperparams['depth']
        args['n_units'] = [500, 300,]
        args['rho'] = 0.3
        args['n_iter'] = self.hyperparams['epochs']
        args['xeta'] = self.hyperparams['lr'] #0.0005
        args['n_batch'] = 100 #500
        self._args = args
				
        dim = self.hyperparams['dimension']
        alpha = self.hyperparams['alpha']
        beta = self.hyperparams['beta']		 
        #self._model = sdne.SDNE(d = dim,
        self._sdne = sdne.SDNE(d = dim,
                                alpha = alpha,
                                beta = beta,
                                **args)
        #self._model.learn_embedding(graph = self.training_data)
        self._sdne.learn_embedding(graph = self.training_data)
        self._model = self._sdne._model
        print("MODEL ", self._model.summary())
        make_keras_pickleable()
        self.fitted = True
        return CallResult(None, True, 1)
						 
						 
    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]:
        #make_keras_pickleable()
        if self.fitted:
            result = self._sdne._Y #produce( )#_Y
        else:
            dim = self.hyperparams['dimension']
            alpha = self.hyperparams['alpha']
            beta = self.hyperparams['beta']		 
            #self._model 
            self._sdne = sdne.SDNE(d = dim,
                                alpha = alpha,
                                beta = beta,
                                **args)
            produce_data, learning_df, nodes_df, edges_df = self._parse_inputs(inputs, return_all = True)
            produce_data = networkx.from_scipy_sparse_matrix(produce_data)
            self._sdne.learn_embedding(graph = produce_data)
            self._model = self._sdne._model
            result = self._sdne._Y
                    
        #result = self._model.learn_embedding(self.training_data)
        #result = result[0]
        
        if self.hyperparams['return_list']:
            result_np = container.ndarray(result, generate_metadata = True)
            return_list = d3m_List([result_np, inputs[1], inputs[2]], generate_metadata = True)        
            return CallResult(return_list, True, 1)
        else:
            result_df = d3m_DataFrame(result, generate_metadata = False)
            
            result_df = result_df.loc[result_df.index.isin(learning_df['d3mIndex'].values)]
            result_df.index.name = 'd3mIndex'



            #result_df.reset_index(drop = False, inplace = True)
            #print("Result post-reset ", result_df)
            #result_df = learning_df.astype(object).join(result_df.astype(object), on = 'd3mIndex')#, on = 'nodeID')
        
            # for column_index in range(result_df.shape[1]):
            #     col_dict = dict(result_df.metadata.query((mbase.ALL_ELEMENTS, column_index)))
            #     #if column_index == 0:
            #     #    col_dict['name'] = 'nodeID'
            #     #else:
            #     col_dict['name'] = 'sdne_' + str(learning_df.shape[1] + column_index)
            #     col_dict['structural_type'] = type(1.0)
            # #    # FIXME: assume we apply corex only once per template, otherwise column names might duplicate
            #     col_dict['semantic_types'] = ('https://metadata.datadrivendiscovery.org/type/Attribute')#, 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/TabularColumn')
            #     result_df.metadata = result_df.metadata.update((mbase.ALL_ELEMENTS,), col_dict)
            

            result_df = d3m_DataFrame(result_df, generate_metadata = True)
            print("Result df (final) ", result_df)
            print("learning _df ", learning_df)
            #output = d3m_DataFrame(result_df, generate_metadata = True, source = self) #index = learning_df['d3mIndex'], 
            #output.index = learning_df.index.copy()
            
            #print('finished output ')
            #self._training_indices = [c for c in learning_df.columns if isinstance(c, str) and 'index' in c.lower()]

            #output = utils.combine_columns(return_result='new', #self.hyperparams['return_result'],
            #                               add_index_columns=True,#self.hyperparams['add_index_columns'], 
            #                               inputs=learning_df, columns_list=[output], source=self, column_indices=self._training_indices)
            #print('SDNE OUTPUT ', output)
            #final_df = result_df
            #final_df.set_index('d3mIndex')
            #return CallResult(final_df, True, 1)
            return CallResult(result_df, True, 1)
            #return CallResult(output, True, 1)

            #append_cols = result_df.loc[learning_df.index]
            #print('APPENDING COLS ', append_cols.shape)
            #print(append_cols.columns)
            #print(append_cols)
            #final_df =  utils.append_columns(learning_df, append_cols)
            #print('FINAL DF ', final_df.shape)
            #print(final_df)
            #result_df.index = [learning_df.index[learning_df[nodeid]==result_df.values[i,0]] for i in range(result_df.values.shape[0])]
            ##result_df.index = learning_df.index.copy()
            #learning_df = utils.append_columns(learning_df, result_df)
            
            # going to return one dataframe (node or edge depending on task)
            
            # add column for node ID ?
            #nodeIDs = inputs[1]
            #result_df['nodeID'] = nodeIDs
            #col_dict = dict(result_df.metadata.query((mbase.ALL_ELEMENTS, column_index)))
            #col_dict['structural_type'] = type(1.0)
            # FIXME: assume we apply corex only once per template, otherwise column names might duplicate                                                            
            #col_dict['name'] = 'corex_' + str(out_df.shape[1] + column_index)
            #col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')

            #corex_df.metadata = corex_df.metadata.update((mbase.ALL_ELEMENTS, column_index), col_dict)
            
            
        
        #inputs[0] = result_np
        
        # TO DO : continue_fit, timeout
        
    
    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, timeout: float = None, iterations: int = None) -> MultiCallResult:
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, timeout : float = None, iterations : int = None) -> MultiCallResult:
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def get_params(self) -> SDNE_Params:
        return SDNE_Params(
            fitted = self.fitted,
            model = self._sdne,
            #model = self._model,
            node_encode = self.node_encode
        )
	
    def set_params(self, *, params: SDNE_Params) -> None:
        self.fitted = params['fitted']
        self._sdne = params['model']
        #self._model = params['model']
        self.node_encode = params['node_encode']
    #def __copy__(self):
    #    new = SDNE()

    #def __deepcopy__(self):
        
