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
        #'feature_construction.graph_transformer.Node2Vec = graphs:Node2Vec',
        'feature_construction.graph_transformer.SDNE = graphs:SDNE'
        #'feature_construction.corex_text.CorexText = corex_text:CorexText',
        #'regression.corex_supervised.EchoLinear = echo_regressor:EchoLinearRegression',
        #'classification.corex_supervised.Echo = echo_sae:EchoClassification',
        #'regression.echo.Echo = echo_sae:EchoRegression'        
    ],
    }

)
