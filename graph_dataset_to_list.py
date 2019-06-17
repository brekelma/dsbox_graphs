import os
import typing
import _config as cfg_

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.container import DataFrame as d3m_DataFrame
from gcn_mix import get_columns_of_type
import numpy as np
import common_primitives

__all__ = ('GraphDatasetToList',)

Inputs = container.Dataset
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    # TO DO: update to take a specified list of DF
    dataframe_resources =  hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Resource ID of a DataFrame to extract if there are multiple tabular resources inside a Dataset and none is a dataset entry point.",
    )


class GraphDatasetToList(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which extracts a DataFrame out of a Dataset.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'dfb8c278-5382-47cd-bd39-f9429890a239',
            'version': '1.0.0',
            'name': "Extract graph tables from Dataset into list of DataFrame",
            'python_path': 'd3m.primitives.data_transformation.graph_to_edge_list.DSBOX',
            'source': {
                'name': 'ISI',
                'contact': 'mailto:brekelma@usc.edu',
                'uris': [
                    'https://github.com/brekelma/dsbox_graphs/graph_dataset_to_list.py'
                    #'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataset_to_dataframe.py',
                    #'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            "installation": [ cfg_.INSTALLATION ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
        )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        try:
            learning_id, learning_df = base_utils.get_tabular_resource(inputs, 'learningData')
        except:
            pass
        
        edge_list = False
        try:
            nodes_id, nodes_df = base_utils.get_tabular_resource(inputs, '0_nodes')
        except:
            try:
                nodes_id, nodes_df = base_utils.get_tabular_resource(inputs, 'nodes')
            except:
                edges_id, edges_df = base_utils.get_tabular_resource(inputs, '1')
                edge_list = True
        if not edge_list:       
            try:
                edges_id, edges_df = base_utils.get_tabular_resource(inputs, '0_edges')
            except:    
                edges_id, edges_df = base_utils.get_tabular_resource(inputs, 'edges')
        
                
        if edge_list:            
            nodes = list(edges_df[[c for c in edges_df.columns if 'source' in c.lower() or 'v1' in c.lower() or 'node1' in c.lower()][0]].astype(np.int32).values)
            nodes.extend(list(edges_df[[c for c in edges_df.columns if 'dest' in c.lower() or 'target' in c.lower() or 'v2' in c.lower() or 'node2' in c.lower()][0]].astype(np.int32).values))
            nodes = np.unique(np.array(nodes))

            node_cols = [col for col in learning_df.columns if 'nodeID' in col or 'attr' in col or 'fea' in col]
            node_col_inds = [learning_df.columns.get_loc(c) for c in node_cols]
            nodes_df = d3m_DataFrame(np.zeros(shape = (nodes.shape[0], len(node_cols))), columns = node_cols, index = nodes)
            nodes_df['nodeID'] = nodes
            

            ldf_vals = learning_df['d3mIndex'].astype(np.int32).values

            


            for j in range(ldf_vals.shape[0]):
                nodes_df.loc[ldf_vals[j]] = learning_df[node_cols].values[j]  





            semantic_types = ('https://metadata.datadrivendiscovery.org/types/Attribute')

            new_meta = metadata_base.DataMetadata()
            learning_df.metadata = new_meta.generate(learning_df)
            new_meta = metadata_base.DataMetadata()
            nodes_df.metadata = new_meta.generate(nodes_df)

            try:
                node_cols.remove('nodeID')
            except:
                pass
            
            for column_index in range(edges_df.shape[1]):

                col_dict = dict(edges_df.metadata.query((metadata_base.ALL_ELEMENTS, column_index)))

                if 'v1' in edges_df.columns[column_index].lower() or 'source' in edges_df.columns[column_index].lower() or 'node1' in edges_df.columns[column_index].lower():
                    col_dict['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/EdgeSource')
                if 'v2' in edges_df.columns[column_index].lower() or 'dest' in edges_df.columns[column_index].lower() or 'target' in edges_df.columns[column_index].lower() or 'node2' in edges_df.columns[column_index].lower():
                    col_dict['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/EdgeTarget')
            
            edges_df.metadata = edges_df.metadata.update((metadata_base.ALL_ELEMENTS, column_index), col_dict)


            for column_index in range(nodes_df.shape[1]):
                col_dict = dict(nodes_df.metadata.query((metadata_base.ALL_ELEMENTS, column_index)))

                if 'attr' in nodes_df.columns[column_index].lower():
                    col_dict['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/Attribute')
                
            nodes_df.metadata = nodes_df.metadata.update((metadata_base.ALL_ELEMENTS, column_index), col_dict)
            
            for column_index in range(learning_df.shape[1]):
                col_dict = dict(learning_df.metadata.query((metadata_base.ALL_ELEMENTS, column_index)))
                #col_dict['structural_type'] = type(1.0)
                # FIXME: assume we apply corex only once per template, otherwise column names might duplicate
                #col_dict['name'] = 'corex_' + str(out_df.shape[1] + column_index)
                if 'attr' in learning_df.columns[column_index].lower():
                    col_dict['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
                
            learning_df.metadata = learning_df.metadata.update((metadata_base.ALL_ELEMENTS, column_index), col_dict)


        if not edge_list:
            learning_df.metadata = self._update_metadata(inputs.metadata, learning_id)
        try:
            nodes_df.metadata = self._update_metadata(inputs.metadata, nodes_id)
        except:
            pass
        edges_df.metadata = self._update_metadata(inputs.metadata, edges_id)

        assert isinstance(learning_df, container.DataFrame), type(learning_df)
        assert isinstance(nodes_df, container.DataFrame), type(nodes_df)
        assert isinstance(edges_df, container.DataFrame), type(edges_df)


        learning_df.reindex(index = learning_df['d3mIndex'])#set_index('d3mIndex')
        return_list = container.List([learning_df, nodes_df, edges_df], generate_metadata = True)
        #return_list = container.List([nodes_df, edges_df], generate_metadata = True)
        return base.CallResult(return_list)

    @classmethod
    def _update_metadata(cls, metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment) -> metadata_base.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update(
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            },
        )

        new_metadata = metadata_base.DataMetadata(resource_metadata)

        new_metadata = metadata.copy_to(new_metadata, (resource_id,))

        # Resource is not anymore an entry point.
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata

    # @classmethod
    # def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]],
    #                hyperparams: Hyperparams) -> typing.Optional[metadata_base.DataMetadata]:
    #     output_metadata = super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

    #     # If structural types didn't match, don't bother.
    #     if output_metadata is None:
    #         return None

    #     if method_name != 'produce':
    #         return output_metadata

    #     if 'inputs' not in arguments:
    #         return output_metadata

    #     inputs_metadata = typing.cast(metadata_base.DataMetadata, arguments['inputs'])

    #     dataframe_resource_id = base_utils.get_tabular_resource_metadata(inputs_metadata, hyperparams['dataframe_resource'])

    #     return cls._update_metadata(inputs_metadata, dataframe_resource_id)
