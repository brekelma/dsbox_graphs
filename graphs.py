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

#CUDA_VISIBLE_DEVICES=""
#Input = container.List
Input = container.DataFrame
#Output = container.List #
Output = container.DataFrame
#container.List #DataFrame #typing.Union[container.DataFrame, None]


class N2V_Params(params.Params):
    fitted: typing.Union[bool, None]

class N2V_Hyperparams(hyperparams.Hyperparams):
    dimension: UniformInt(
        lower = 10, 
        upper = 100,
        default = 50,
        #q = 5, 
        description = 'dimension of latent embedding',
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
)
    walk_len: UniformInt(
        lower = 1, 
        upper = 10,
        default = 5,
        #q = 1, 
        description = 'length of random walk',
        semantic_types=["http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
)

    #num_walks: typing.Union[int, None]
    #context_size: typing.Union[int, None]
    #return_weight: typing.Union[float, int, None]
    #inout_weight: typing.Union[float, int, None]

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
#class
						 
						 
class SDNE_Params(params.Params):
    fitted: typing.Union[bool, None]
    model: typing.Union[sdne.SDNE, None]
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
        default = 100,
        #q = 5,
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

class SDNE(UnsupervisedLearnerPrimitiveBase[Input, Output, SDNE_Params, SDNE_Hyperparams]):
    """
    Graph embedding method
    """

    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "",
        "version": "1.0.0",
        "name": "SDNE",
        "description": "graph embedding",
        "python_path": "d3m.primitives.feature_construction.graph_transformer.SDNE",
        "original_python_path": "gem.gem",
        "source": {
            "name": "ISI",
            "contact": "mailto:brekelma@usc.edu",
            "uris": [ "https://github.com/brekelma/dsbox_graphs" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": ["EXPECTATION_MAXIMIZATION_ALGORITHM", "LATENT_DIRICHLET_ALLOCATION"],
        "primitive_family": "FEATURE_CONSTRUCTION",
        "hyperparams_to_tune": ["dimension", "beta", "alpha"]
    })
    
    def __init__(self, *, hyperparams : SDNE_Hyperparams) -> None:
        super(SDNE, self).__init__(hyperparams = hyperparams)
        # nu1 = 1e-6, nu2=1e-6, K=3,n_units=[500, 300,], rho=0.3, n_iter=30, xeta=0.001,n_batch=500
	
    def set_training_data(self, *, inputs : Input) -> None:
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
        
        if self.hyperparams['return_list']:
            result_np = container.ndarray(result, generate_metadata = True)
            return_list = d3m_List([result_np, inputs[1], inputs[2]], generate_metadata = True)        
            return CallResult(return_list, True, 1)
        else:
            result_df = d3m_DataFrame(result, generate_metadata = True)
            nodeIDs = inputs[1]
            result_df['nodeID'] = nodeIDs
            result_df = d3m_DataFrame(result_df, generate_metadata = True)
            #col_dict = dict(result_df.metadata.query((mbase.ALL_ELEMENTS, column_index)))
            #col_dict['structural_type'] = type(1.0)
            # FIXME: assume we apply corex only once per template, otherwise column names might duplicate                                                            
            #col_dict['name'] = 'corex_' + str(out_df.shape[1] + column_index)
            #col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')

            #corex_df.metadata = corex_df.metadata.update((mbase.ALL_ELEMENTS, column_index), col_dict)
            
            return CallResult(result_df, True, 1)
        
        #inputs[0] = result_np
        
        # TO DO : continue_fit, timeout
        
    
    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, timeout: float = None, iterations: int = None) -> MultiCallResult:
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Input, timeout : float = None, iterations : int = None) -> MultiCallResult:
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def get_params(self) -> SDNE_Params:
        return SDNE_Params(
            fitted = self.fitted,
            model = self._model)
	
    def set_params(self, *, params: SDNE_Params) -> None:
        self.fitted = params['fitted']
        self._model = params['model']

    #def __copy__(self):
    #    new = SDNE()

    #def __deepcopy__(self):
        
