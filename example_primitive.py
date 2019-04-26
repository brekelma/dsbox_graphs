import os
import sys
import typing
import networkx

import tensorflow as tf
#from GEM.gem.embedding import node2vec
from GEM.gem.embedding import sdne
#from GEM.gem.embedding import sdne_utils
import keras.models
import tempfile

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

import config as cfg_



Input = container.DataFrame
Output = container.DataFrame
                 
                         
class SDNE_Params(params.Params):

    ''' 
    Attributes necessary to resume training or run on test data (if loaded from pickle)

    Code specifications of parameters: 
        https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/params.py
    '''

    fitted: typing.Union[bool, None] # fitted required, set once primitive is trained
    model: typing.Union[sdne.SDNE, None]

class SDNE_Hyperparams(hyperparams.Hyperparams):

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



class SDNE(UnsupervisedLearnerPrimitiveBase[Input, Output, SDNE_Params, SDNE_Hyperparams]):
    """
    See base classes here : 
        https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/primitive_interfaces

    """

    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "",
        "version": "1.0.0",
        "name": "SDNE",
        "description": "graph embedding",
        # ask about naming convention
        "python_path": "d3m.primitives.feature_construction.graph_transformer.SDNE",
        "original_python_path": "gem.gem",
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
    
    def __init__(self, *, hyperparams : SDNE_Hyperparams) -> None:
        super(SDNE, self).__init__(hyperparams = hyperparams)

	
    def set_training_data(self, *, inputs : Input) -> None:
        # Will change...

        # input should be dataframes of nodes, edges

        G = inputs[0].copy()
        if type(G) == networkx.classes.graph.Graph:
            if networkx.is_weighted(G):
                E = int(networkx.number_of_edges(G))
                #g = self._pass_to_ranks(G, nedges = E)
            else:
                E = int(networkx.number_of_edges(G))
                g = networkx.to_numpy_array(G)
        elif type(G) is np.ndarray:
            G = networkx.to_networkx_graph(G)
            E = int(networkx.number_of_edges(G))
            #g = self._pass_to_ranks(G, nedges = E)
        else:
            raise ValueError("networkx Graph and n x d numpy arrays only")

        self.training_data = G
        #self.training_data = inputs
        self.fitted = False

    def fit(self, *, timeout : float = None, iterations : int = None) -> None:
        
        make_keras_pickleable()
        
        if self.fitted:
            return CallResult(None, True, 1)

        # ******************************************
        # Feel free to fill in less important hyperparameters or example architectures here
        # ******************************************

        args = {}
        args['nu1'] = 1e-6
        args['nu2'] = 1e-6
        args['K'] = 3
        args['n_units'] = [500, 300,]
        args['rho'] = 0.3
        args['n_iter'] = 2
        args['xeta'] = 0.001
        args['n_batch'] = 100 #500
        self._args = args
				
        dim = self.hyperparams['dimension']
        alpha = self.hyperparams['alpha']
        beta = self.hyperparams['beta']		 
        self._model = sdne.SDNE(d = dim,
                                alpha = alpha,
                                beta = beta,
                                **args)
        print()
        print("***************")
        print(self.training_data)
        print(type(self.training_data))
        print("***************")
        self._model.learn_embedding(self.training_data)
        
        self.fitted = True
        return CallResult(None, True, 1)
						 
						 
    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]:
        if self.fitted:
            result = self._model._Y
        else:
            dim = self.hyperparams['dimension']
            alpha = self.hyperparams['alpha']
            beta = self.hyperparams['beta']		 
            self._model = sdne.SDNE(d = dim,
                                alpha = alpha,
                                beta = beta,
                                **args)
            
            
        result = self._model.learn_embedding(self.training_data)
        result = result[0]
        
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

    def get_params(self) -> SDNE_Params:

        # fill in with model attributes 

        return SDNE_Params(
            fitted = self.fitted,
            model = self._model)
	
    def set_params(self, *, params: SDNE_Params) -> None:

        # assign model attributes (e.g. in loading from pickle)

        self.fitted = params['fitted']
        self._model = params['model']
