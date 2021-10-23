import dgl
import torch as th
import os,sys,math,glob,random,ROOT
import numpy as np
import argparse

import options


def main(argv):

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")
    parser.add_argument("-n", "--ntuples", type=str, required=True, dest="ntuples", help="names of ntuples to combine")   

    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name
    ntuples = args.ntuples.split()
    ntuples.sort()

    #import options from option file
    valp = options.valp
    testp = options.testp

    paramfile_name = data_path+data_name+"_params"
    train_outfile_name = data_path+data_name+"_train.bin"
    val_outfile_name = data_path+data_name+"_val.bin"
    test_outfile_name = data_path+data_name+"_test.bin"

    g_list = []
    ngraphs = 0
    total_cut = total_remain = 0

    for ntuple in ntuples:
        infile_name = data_path+ntuple+".bin"
        graphs = dgl.load_graphs(infile_name)[0]
        ngraphs += len(graphs)

        #add file number to graphs
        for graph in graphs:
            ntracks = graph.num_nodes()
            total_cut += int(th.sum(graph.ndata['passed_cuts'] == 0))
            total_remain += int(th.sum(graph.ndata['passed_cuts'] == 1))
        
            g_list.append(graph)

    random.shuffle(g_list)

    #calculate size of testing, training and validation set
    test_len = int(round(testp*ngraphs))
    val_len = int(round(valp*ngraphs))
    train_len = int(ngraphs - (test_len + val_len))

    print("Cut {}% of {} tracks".format(100*total_cut/(total_cut+total_remain), total_cut+total_remain))

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
