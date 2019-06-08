from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="dsbox_graphs",
    version="1.0.0",
    description="Graph Embedding and Convolution primitives",
    license="AGPL-3.0",
    author="Rob Brekelmans",
    author_email="brekelma@usc.edu",
    keywords='d3m_primitive',
    packages=find_packages(),
    url='https://github.com/brekelma/dsbox_graphs',
    download_url='https://github.com/brekelma/dsbox_graphs',
    install_requires=[],
    long_description=long_description,
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
    ], 
    entry_points = {
    'd3m.primitives': [
        'feature_construction.sdne.DSBOX = sdne:SDNE',
        'data_transformation.graph_to_edge_list.DSBOX = graph_dataset_to_list:GraphDatasetToList',
        'feature_construction.gcn_mixhop.DSBOX = gcn_mix:GCN'
        #'feature_construction.graph_transformer.SDNE = sdne:SDNE',
	#'data_transformation.graph_to_edge_list.DSBOX = graph_dataset_to_list:GraphDatasetToList',
	#'feature_construction.graph_transformer.GCN = gcn_mix:GCN'
    ],
    }

)
