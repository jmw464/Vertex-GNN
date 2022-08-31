#!/usr/bin/env python

####################################### prune_graphs.py #######################################
# PURPOSE: cut marked tracks from test/train/val graph files
# EDIT TO: update definition of edge labels (if modified in create_graphs)
# -------------------------------------------Summary-------------------------------------------
# This script is run as the final step in the processing chain and it requires DGL graph files
# split into test/train/val dataset as input. The purpose of this script is to remove tracks
# marked to be cut from all graphs in each dataset. Because jet-level cuts are performed when
# the graphs are initially created, this script does not change the number of graphs in each
# file (jets with fewer than 2 passing tracks are also removed beforehand). Additionally, this
# script outputs a "paramfile" containing information about the relative prevalence of each
# type of edge label in the training dataset, allowing the GNN to re-weight labels during
# training.
###############################################################################################


import dgl
import torch as th
import os,sys,math,glob,random,ROOT
import numpy as np
import argparse


def prune_graph(graph):
    nodes_to_cut = []
    cut_array = graph.ndata['passed_cuts']
    for i in range(len(cut_array)):
        if cut_array[i] == 0:
            nodes_to_cut.append(i)
    graph.remove_nodes(nodes_to_cut,store_ids=False)
    return graph


def main(argv):

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")

    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name

    infile_prefix = data_path+data_name+"_"
    total_true = total_edges = total_b = total_c = 0

    dataset_len = np.zeros(3)
    for i, dataset_type in enumerate(['train', 'val', 'test']):
        g_list = []
        graphs = dgl.load_graphs(infile_prefix+dataset_type+".bin")[0]
        dataset_len[i] = len(graphs)

        for graph in graphs:
            ntracks = graph.num_nodes()
            graph = prune_graph(graph)
            
            #only consider training dataset when writing values to paramfile
            if dataset_type == 'train':
                total_true += int(th.sum(graph.edata['bin_labels'][:,0]))
                total_b += int(th.sum(graph.edata['mult_labels'][:,0] == 1))
                total_c += int(th.sum(graph.edata['mult_labels'][:,0] == 2))
                total_edges += list(graph.edata['bin_labels'][:,0].size())[0]
            
            g_list.append(graph)

        random.shuffle(g_list)
        dgl.save_graphs(infile_prefix+dataset_type+'.pruned.bin', g_list)
   
    #store important values in paramfile
    paramfile = open(infile_prefix+'params', "w")
    paramfile.write(str(dataset_len[0])+'\n') #train length
    paramfile.write(str(dataset_len[1])+'\n') #val length
    paramfile.write(str(dataset_len[2])+'\n') #test length
    paramfile.write(str(total_true/total_edges)+'\n')
    paramfile.write(str(total_b/total_edges)+'\n')
    paramfile.write(str(total_c/total_edges)+'\n')
    paramfile.close()


if __name__ == '__main__':
    main(sys.argv)
