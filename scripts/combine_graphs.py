import dgl
import torch as th
import os,sys,math,glob,random,ROOT
import numpy as np
import argparse

#############################################SCRIPT PARAMS#################################################

#output data parameters
valp = 0.2 #fraction of data used for validation
testp = 0.1 #fraction of data used for testing

###########################################################################################################

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

    paramfile_name = data_path+data_name+"_params"
    train_outfile_name = data_path+data_name+"_train.bin"
    val_outfile_name = data_path+data_name+"_val.bin"
    test_outfile_name = data_path+data_name+"_test.bin"

    g_list = []
    ngraphs = ifile = 0
    total_true = total_edges = total_b = total_c = total_btoc = total_o = 0

    for ntuple in ntuples:
        infile_name = data_path+data_name+"_"+ntuple+".bin"
        graphs = dgl.load_graphs(infile_name)[0]
        ngraphs += len(graphs)

        #add file number to graphs
        for graph in graphs:
            ntracks = graph.num_nodes()
            total_true += int(th.sum(graph.edata['bin_labels'][:,0]))
            total_b += int(th.sum(graph.edata['mult_labels'][:,0] == 1))
            total_c += int(th.sum(graph.edata['mult_labels'][:,0] == 2))
            total_btoc += int(th.sum(graph.edata['mult_labels'][:,0] == 3))
            total_o += int(th.sum(graph.edata['mult_labels'][:,0] == 4))
            total_edges += list(graph.edata['bin_labels'][:,0].size())[0]
        
            for i in range(ntracks):
                graph.ndata['info'][i,0] = ifile

            g_list.append(graph)

        ifile += 1        
    
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

    #store important values in paramfile
    paramfile = open(paramfile_name, "w")
    paramfile.write(str(test_len)+'\n')
    paramfile.write(str(val_len)+'\n')
    paramfile.write(str(train_len)+'\n')
    paramfile.write(str(total_true/total_edges)+'\n')
    paramfile.write(str(total_b/total_edges)+'\n')
    paramfile.write(str(total_c/total_edges)+'\n')
    paramfile.write(str(total_btoc/total_edges)+'\n')
    paramfile.write(str(total_o/total_edges)+'\n')
    paramfile.close()


if __name__ == '__main__':
    main(sys.argv)
