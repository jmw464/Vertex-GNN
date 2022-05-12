#!/usr/bin/env python

###################################### combine_graphs.py ######################################
# PURPOSE: combine different DGL graph files into one, split data into test/train/val
# EDIT TO: /
# -------------------------------------------Summary-------------------------------------------
# This script is run on binary graph files created with "create_graphs.py". Its purpose is
# two-fold: combining multiple different graph files into a single GNN dataset (such as when
# more than one ntuple is required) as well as splitting the combined dataset into testing,
# training and validation data. The reason this is done at this stage is that all data must be
# normalized only with respect to the training feature distributions. This script is written
# to be very general, there should be no reason to edit it.
###############################################################################################


import dgl
import torch as th
import os,sys,math,glob,random,ROOT
import numpy as np
import argparse


def main(argv):

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")
    parser.add_argument("-n", "--ntuples", type=str, required=True, dest="ntuples", help="names of ntuples to combine")   
    parser.add_argument("-f", "--options", type=str, required=True, dest="option_file", help="name of file containing script options")

    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name
    option_file = args.option_file
    ntuples = args.ntuples.split()
    ntuples.sort()

    options = __import__(option_file, globals(), locals(), [], 0)

    #import options from option file
    valp = options.valp
    testp = options.testp

    paramfile_name = data_path+data_name+"_params"
    train_outfile_name = data_path+data_name+"_train.bin"
    val_outfile_name = data_path+data_name+"_val.bin"
    test_outfile_name = data_path+data_name+"_test.bin"

    g_list = []
    ngraphs = 0
    for ntuple in ntuples:
        infile_names = glob.glob(data_path+ntuple+"_*.bin")
        for infile_name in infile_names:
            graphs = dgl.load_graphs(infile_name)[0]
            g_list.extend(graphs)
            ngraphs += len(graphs)

    random.shuffle(g_list)

    #calculate size of testing, training and validation set
    test_len = int(round(testp*ngraphs))
    val_len = int(round(valp*ngraphs))
    train_len = int(ngraphs - (test_len + val_len))

    #split g_list
    test_list = g_list[:test_len]
    val_list = g_list[test_len:test_len+val_len]
    train_list = g_list[test_len+val_len:]

    #save graphs to file
    dgl.save_graphs(test_outfile_name, test_list)
    dgl.save_graphs(val_outfile_name, val_list)
    dgl.save_graphs(train_outfile_name, train_list)


if __name__ == '__main__':
    main(sys.argv)
