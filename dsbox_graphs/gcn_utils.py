import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from d3m.metadata.base import Metadata, DataMetadata, SelectorSegment, ALL_ELEMENTS
from d3m.base import utils as base_utils
import d3m.container as container
from d3m.metadata.base import CONTAINER_SCHEMA_VERSION, DataMetadata, ALL_ELEMENTS, SelectorSegment

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

def assign_scattered(inputs, add_to_zeros=False):
        # "Undo" slice.  Used on loss function to give calculated loss for supervised examples, else 0 
        # inputs = [loss_on_slices, shape_ref, indices]
        slice_loss = inputs[0]
        shape_ref = inputs[1]
        # e.g. loss goes in batch dim 0,2,4,6,8, inds.shape = (5,1)
        inds = tf.cast(inputs[-1], tf.int32) #tf.expand_dims(, -1)
        inds = tf.expand_dims(inds, -1)

        # scatter_nd defaults to zeros
        # full_loss = tf.scatter_nd(inds, 
        #                         slice_loss, 
        #                         shape = [tf.shape(input=shape_ref)[0]])
        # import IPython
        # IPython.embed()
        # full_loss = tf.tensor_scatter_nd_add(tf.zeros(tf.shape(shape_ref), slice_loss.dtype), inds, slice_loss)

        if add_to_zeros:
                full_loss = tf.scatter_nd(inds, 
                        slice_loss, 
                        tf.shape(shape_ref))
        else:
                full_loss = tf.tensor_scatter_nd_add(
                        shape_ref, inds, slice_loss, name='full_zeros_output'
                        )

        return full_loss #tf.reshape(full_loss, (-1,))


def import_loss(inputs, function = None, first = None, flatten=False):
        if isinstance(function, str):
                        import importlib
                        mod = importlib.import_module('keras.losses')
                        function = getattr(mod, function)
        #try:
        if not flatten:
                return function(inputs[0], inputs[-1])
        else:
                input_0 = tf.reshape(inputs[0], [-1,1])
                input_1 = tf.reshape(inputs[-1], [-1,1])
                f = tf.reshape(function(input_0, input_1), tf.shape(inputs[0]))
                return f  #if function is not None else inputs
        #return function(inputs[0], inputs[-1]) if function is not None else inputs
        #except:
        #                inputs[0] = tf.gather(inputs[0], np.arange(first))
        #                return function(inputs[0], inputs[-1]) if function is not None else inputs


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

#     print()
#     print()
#     print("LIST COLUMNS to USE (not of type)")
#     print(columns_to_use)
#     print(columns_not_to_use)
#     print()
#     print()

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
