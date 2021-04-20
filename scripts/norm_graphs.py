import dgl
import torch as th
import os,sys,math,glob,ROOT
import numpy as np

#############################################SCRIPT PARAMS#################################################

#input data parameters
file_path = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/"
file_name = "Btag_07_19_cut"

###########################################################################################################

train_infile_name = file_path+file_name+"_train.bin"
val_infile_name = file_path+file_name+"_val.bin"
test_infile_name = file_path+file_name+"_test.bin"
train_outfile_name = file_path+file_name+"_train.normed.bin"
val_outfile_name = file_path+file_name+"_val.normed.bin"
test_outfile_name = file_path+file_name+"_test.normed.bin"


def main(argv):
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
    mean_features[2] = math.pi/2. #track theta varies from 0 to pi
    std_features[2] = math.pi/2.
    mean_features[3] = 0 #track phi varies from -pi to pi
    std_features[3] = math.pi
    mean_features[6] = 0 #q is either -1 or 1
    std_features[6] = 1
    mean_features[9] = 0 #jet phi varies from -pi to pi
    std_features[9] = math.pi

    #apply normalization from training data to all graph features
    print("Normalizing training data")
    for graph in train_graphs:
        features = graph.ndata['features'].numpy()
        normed_features = np.divide(features-mean_features, std_features)
        graph.ndata['features'] = th.from_numpy(normed_features)
    dgl.save_graphs(train_outfile_name, train_graphs)

    print("Normalizing validation data")
    val_graphs = dgl.load_graphs(val_infile_name)[0]
    for graph in val_graphs:
        features = graph.ndata['features'].numpy()
        normed_features = np.divide(features-mean_features, std_features)
        graph.ndata['features'] = th.from_numpy(normed_features)
    dgl.save_graphs(val_outfile_name, val_graphs)

    print("Normalizing testing data")
    test_graphs = dgl.load_graphs(test_infile_name)[0]
    for graph in test_graphs:
        features = graph.ndata['features'].numpy()
        normed_features = np.divide(features-mean_features, std_features)
        graph.ndata['features'] = th.from_numpy(normed_features)
    dgl.save_graphs(test_outfile_name, test_graphs)


if __name__ == '__main__':
    main(sys.argv)
