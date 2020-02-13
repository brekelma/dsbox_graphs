import os
import sys
import json
import typing
import numpy as np
import pandas as pd
import networkx as nx

from d3m.primitive_interfaces import base, transformer

from d3m import container



def _jhu_load(inputs : container.Dataset):
    data_resources_keys = list(inputs.keys())
    
    # obtain the path to dataset
    temp_json = inputs.to_json_structure()
    datasetDoc_uri = temp_json['location_uris'][0][7:]
    location_base_uri = '/'.join(datasetDoc_uri.split('/')[:-1])
    
    with open(datasetDoc_uri) as json_file:
        datasetDoc_json = json.load(json_file)
        dataResources = datasetDoc_json['dataResources']
        
    # get the task type from the task docs
    temp_path = datasetDoc_uri.split('/')
    problemDoc_uri = '/'.join(temp_path[:-2]) + '/' + '/'.join(temp_path[-2:]).replace('dataset', 'problem')
    
    with open(problemDoc_uri) as json_file:
        task_types = json.load(json_file)['about']['taskKeywords']
            
    # TODO consider avoiding explicit use of problem type throughout pipeline
    TASK = "" 
    for task in task_types:
        if task in ["communityDetection", "linkPrediction", "vertexClassification", "graphMatching"]:
            TASK = task
    if TASK == "":
        raise exceptions.NotSupportedError("only graph tasks are supported")

    # load the graphs and convert to a networkx object
    graphs = []
    nodeIDs = []
    for i in dataResources:
        with open("sdne_log.csv",'a') as f:
            f.write("\n")
            f.write(str(i['resType']))
        if i['resType'] == "table":
            df = inputs['learningData']
        elif i['resType'] == 'graph':
            graph_temp = nx.read_gml(location_base_uri + "/" + i['resPath'])
            graphs.append(graph_temp)
            if TASK in ["communityDetection", "vertexClassification"]:
                nodeIDs_temp = list(nx.get_node_attributes(graphs[0], 'nodeID').values())
                nodeIDs_temp = np.array([str(i) for i in nodeIDs_temp])
                nodeIDs_temp = container.ndarray(nodeIDs_temp)
                nodeIDs.append(nodeIDs_temp)
        elif i['resType'] == "edgeList":
            temp_graph = _read_edgelist(
                location_base_uri + "/" + i['resPath'],
                i["columns"], )
            graphs.append(temp_graph)
            if TASK in ["communityDetection", "vertexClassification"]:
                nodeIDs_temp = list(temp_graph.nodes)
                nodeIDs_temp = np.array([str(i) for i in nodeIDs_temp])
                nodeIDs_temp = container.ndarray(nodeIDs_temp)
                nodeIDs.append(nodeIDs_temp)

    return df, graphs, nodeIDs, TASK #CallResult(container.List([df, graphs, nodeIDs, TASK]))
                    
        
def _read_edgelist(path, columns):
    # assumed that any edgelist passed has a source in the first col
    # and a reciever in the second col.
    # TODO make this function handle time series (Ground Truth)
    # specify columns of edges
    from_column = columns[1]['colName']
    to_column = columns[2]['colName']
    # specify types
    dtypes_dict = {from_column: str, to_column: str}
    edgeList=pd.read_csv(path, dtype=dtypes_dict)
    G = nx.convert_matrix.from_pandas_edgelist(edgeList,
                                               from_column,
                                               to_column)
    return G
