import os
import typing

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

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
            'id': '4b42ce1e-9b98-4a25-b68e-fad13311eb65',
            'version': '0.3.0',
            'name': "Extract graph tables from Dataset into list of DataFrame",
            'python_path': 'd3m.primitives.data_transformation.graph_to_edge_list.DSBOX',
            'source': {
                'name': 'Rob Brekelmans',
                'contact': 'mailto:brekelma@usc.edu',
                'uris': [
                    'https://gitlab.com/brekelma/dsbox_graphs/graph_dataset_to_list.py'
                    #'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataset_to_dataframe.py',
                    #'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/brekelma/dsbox_graphs.git@{git_commit}#egg=dsbox_graphs'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
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
        print('resources ?? ', inputs.keys())
        
        try:
            nodes_id, nodes_df = base_utils.get_tabular_resource(inputs, '0_nodes')
        except:
            nodes_id, nodes_df = base_utils.get_tabular_resource(inputs, 'nodes')
        try:
            edges_id, edges_df = base_utils.get_tabular_resource(inputs, '0_edges')
        except:
            edges_id, edges_df = base_utils.get_tabular_resource(inputs, 'edges')

        learning_df.metadata = self._update_metadata(inputs.metadata, learning_id)
        nodes_df.metadata = self._update_metadata(inputs.metadata, nodes_id)
        edges_df.metadata = self._update_metadata(inputs.metadata, edges_id)

        assert isinstance(learning_df, container.DataFrame), type(learning_df)
        assert isinstance(nodes_df, container.DataFrame), type(nodes_df)
        assert isinstance(edges_df, container.DataFrame), type(edges_df)

        #learning_df.index.name = 'd3mIndex'
        print('learning df ')
        print(learning_df)
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
