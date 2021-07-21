import dgl
import torch as th
import os,sys,math,glob,ROOT
import numpy as np
import argparse

import options


def main(argv):

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")
    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name

    #import options from option file
    nnfeatures_base = options.nnfeatures_base 
    nnfeatures_errors = options.nnfeatures_errors
    nnfeatures_corrs = options.nnfeatures_corrs
    nnfeatures_hits = options.nnfeatures_hits
    incl_errors = options.incl_errors
    incl_corr = options.incl_corr
    incl_hits = options.incl_hits

    train_infile_name = data_path+data_name+"_train.bin"
    val_infile_name = data_path+data_name+"_val.bin"
    test_infile_name = data_path+data_name+"_test.bin"
    train_outfile_name = data_path+data_name+"_train.normed.bin"
    val_outfile_name = data_path+data_name+"_val.normed.bin"
    test_outfile_name = data_path+data_name+"_test.normed.bin"
    normfile_name = data_path+data_name+"_norm"

    train_graphs = dgl.load_graphs(train_infile_name)[0]
    num_features = train_graphs[0].ndata['features'].size()[1]
    mean_features = np.zeros(num_features)
    std_features = np.zeros(num_features)

    print("Calculating mean of features")
    #calculate mean of training features
    total_tracks = 0
    for graph in train_graphs:
        features = graph.ndata['features'].numpy()
        mean_features += np.sum(features,axis=0)
        total_tracks += graph.ndata['features'].size()[0]
    mean_features = mean_features/total_tracks

    print("Calculating STD of features")
    #calculate std of training features
    for graph in train_graphs:
        features = graph.ndata['features'].numpy()
        std_features += np.sum(np.square(features-mean_features),axis=0)
    std_features = np.sqrt(std_features/total_tracks)

    #manually set normalization parameters for special features (features that have a fixed range are set to vary from -1 to 1)
    mean_features[1] = math.pi/2. #track theta varies from 0 to pi
    std_features[1] = math.pi/2.
    mean_features[2] = 0 #track phi varies from -pi to pi
    std_features[2] = math.pi
    mean_features[7] = 0 #jet phi varies from -pi to pi
    std_features[7] = math.pi
    offset = nnfeatures_base
    if incl_errors:
        mean_features[offset] = 0
        std_features[offset] = std_features[0]**2
        mean_features[offset+1] = 0
        std_features[offset+1] = std_features[1]**2
        mean_features[offset+2] = 0
        std_features[offset+2] = std_features[2]**2
        mean_features[offset+3] = 0
        std_features[offset+3] = std_features[3]**2
        mean_features[offset+4] = 0
        std_features[offset+4] = std_features[4]**2
        offset += nnfeatures_errors
    if incl_corr:
        mean_features[offset] = 0
        std_features[offset] = std_features[3]*std_features[4]
        mean_features[offset+1] = 0
        std_features[offset+1] = std_features[3]*std_features[2]
        mean_features[offset+2] = 0
        std_features[offset+2] = std_features[3]*std_features[1]
        mean_features[offset+3] = 0
        std_features[offset+3] = std_features[3]*std_features[0]
        mean_features[offset+4] = 0
        std_features[offset+4] = std_features[4]*std_features[2]
        mean_features[offset+5] = 0
        std_features[offset+5] = std_features[4]*std_features[1]
        mean_features[offset+6] = 0
        std_features[offset+6] = std_features[4]*std_features[0]
        mean_features[offset+7] = 0
        std_features[offset+7] = std_features[2]*std_features[1]
        mean_features[offset+8] = 0
        std_features[offset+8] = std_features[2]*std_features[0]
        mean_features[offset+9] = 0
        std_features[offset+9] = std_features[1]*std_features[0]

    #correct 0 std if necessary
    for i in range(num_features):
        if std_features[i] == 0:
            std_features[i] == 1
    
    #store normalization parameters in file
    normfile = open(normfile_name, "w")
    for i in range(len(mean_features)):
        normfile.write(str(mean_features[i])+'\n')
        normfile.write(str(std_features[i])+'\n')
    normfile.close()

    #apply normalization from training data to all graph features
    print("Normalizing {} training graphs".format(len(train_graphs)))
    for graph in train_graphs:
        features = graph.ndata['features'].numpy()
        normed_features = np.divide(features-mean_features, std_features)
        graph.ndata['features'] = th.from_numpy(normed_features)
    dgl.save_graphs(train_outfile_name, train_graphs)

    val_graphs = dgl.load_graphs(val_infile_name)[0]
    print("Normalizing {} validation graphs".format(len(val_graphs)))
    for graph in val_graphs:
        features = graph.ndata['features'].numpy()
        normed_features = np.divide(features-mean_features, std_features)
        graph.ndata['features'] = th.from_numpy(normed_features)
    dgl.save_graphs(val_outfile_name, val_graphs)

    test_graphs = dgl.load_graphs(test_infile_name)[0]
    print("Normalizing {} testing graphs".format(len(test_graphs)))
    for graph in test_graphs:
        features = graph.ndata['features'].numpy()
        normed_features = np.divide(features-mean_features, std_features)
        graph.ndata['features'] = th.from_numpy(normed_features)
    dgl.save_graphs(test_outfile_name, test_graphs)


if __name__ == '__main__':
    main(sys.argv)
