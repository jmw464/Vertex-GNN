import dgl
import torch as th
import os,sys,math,glob,ROOT
import numpy as np
import argparse


def main(argv):

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")
    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name

    train_infile_name = data_path+data_name+"_train.bin"
    val_infile_name = data_path+data_name+"_val.bin"
    test_infile_name = data_path+data_name+"_test.bin"
    train_outfile_name = data_path+data_name+"_train.normed.bin"
    val_outfile_name = data_path+data_name+"_val.normed.bin"
    test_outfile_name = data_path+data_name+"_test.normed.bin"

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
    mean_features[2] = 0 #theta error varies with theta
    std_features[2] = 1#std_features[1]
    mean_features[3] = 0 #track phi varies from -pi to pi
    std_features[3] = math.pi
    mean_features[4] = 0 #phi error varies with phi
    std_features[4] = 1#std_features[3]
    mean_features[6] = 0 #d0 error varies with d0
    std_features[6] = 1#std_features[5]
    mean_features[8] = 0 #z0 error varies with z0
    std_features[8] = 1#std_features[7]
    mean_features[9] = 0 #q is either -1 or 1
    std_features[9] = 1
    mean_features[12] = 0 #jet phi varies from -pi to pi
    std_features[12] = math.pi

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
