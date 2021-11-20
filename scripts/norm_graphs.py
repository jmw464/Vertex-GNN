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

    train_infile_name = data_path+data_name+"_train.bin"
    val_infile_name = data_path+data_name+"_val.bin"
    test_infile_name = data_path+data_name+"_test.bin"
    train_outfile_name = data_path+data_name+"_train.normed.bin"
    val_outfile_name = data_path+data_name+"_val.normed.bin"
    test_outfile_name = data_path+data_name+"_test.normed.bin"
    normfile_name = data_path+data_name+"_norm"

    incl_errors = incl_hits = incl_corr = incl_vweight = False
    train_graphs = dgl.load_graphs(train_infile_name)[0]
    num_features_base = train_graphs[0].ndata['features_base'].size()[1]
    mean_features_base = np.zeros(num_features_base)
    std_features_base = np.zeros(num_features_base)
    if 'features_vweight' in train_graphs[0].ndata.keys():
        incl_vweight = True
        num_features_vweight = train_graphs[0].ndata['features_vweight'].size()[1]
        mean_features_vweight = np.zeros(num_features_vweight)
        std_features_vweight = np.zeros(num_features_vweight)
    if 'features_errors' in train_graphs[0].ndata.keys():
        incl_errors = True
        num_features_errors = train_graphs[0].ndata['features_errors'].size()[1]
        mean_features_errors = np.zeros(num_features_errors)
        std_features_errors = np.zeros(num_features_errors)
    if 'features_hits' in train_graphs[0].ndata.keys():
        incl_hits = True
        num_features_hits = train_graphs[0].ndata['features_hits'].size()[1]
        mean_features_hits = np.zeros(num_features_hits)
        std_features_hits = np.zeros(num_features_hits)
    if 'features_corr' in train_graphs[0].ndata.keys():
        incl_corr = True
        num_features_corr = train_graphs[0].ndata['features_corr'].size()[1]
        mean_features_corr = np.zeros(num_features_corr)
        std_features_corr = np.zeros(num_features_corr)

    print("Calculating mean of features")
    #calculate mean of training features - error and cov scaling is based only on scaling of base features
    total_tracks = 0
    for graph in train_graphs:
        features_base = graph.ndata['features_base'].numpy()
        mean_features_base += np.sum(features_base,axis=0)
        if incl_vweight:
            features_vweight = graph.ndata['features_vweight'].numpy()
            mean_features_vweight += np.sum(features_vweight,axis=0)
        if incl_hits:
            features_hits = graph.ndata['features_hits'].numpy()
            mean_features_hits += np.sum(features_hits,axis=0)
        total_tracks += graph.ndata['features_base'].size()[0]
    mean_features_base = mean_features_base/total_tracks
    if incl_vweight: mean_features_vweight = mean_features_vweight/total_tracks
    if incl_hits: mean_features_hits = mean_features_hits/total_tracks

    print("Calculating STD of features")
    #calculate std of training features
    for graph in train_graphs:
        features_base = graph.ndata['features_base'].numpy()
        std_features_base += np.sum(np.square(features_base-mean_features_base),axis=0)
        if incl_vweight:
            features_vweight = graph.ndata['features_vweight'].numpy()
            std_features_vweight += np.sum(np.square(features_vweight-mean_features_vweight),axis=0)
        if incl_hits:
            features_hits = graph.ndata['features_hits'].numpy()
            std_features_hits += np.sum(np.square(features_hits-mean_features_hits),axis=0)
    std_features_base = np.sqrt(std_features_base/total_tracks)
    if incl_vweight: std_features_vweight = np.sqrt(std_features_vweight/total_tracks)
    if incl_hits: std_features_hits = np.sqrt(std_features_hits/total_tracks)

    #manually set normalization parameters for special features (features that have a fixed range are set to vary from -1 to 1)
    mean_features_base[1] = math.pi/2. #track theta varies from 0 to pi
    std_features_base[1] = math.pi/2.
    mean_features_base[2] = 0 #track phi varies from -pi to pi
    std_features_base[2] = math.pi
    mean_features_base[7] = 0 #jet phi varies from -pi to pi
    std_features_base[7] = math.pi
    if incl_errors: #take std of variance to be std of features squared
        std_features_errors[0] = std_features_base[0]
        std_features_errors[1] = std_features_base[1]
        std_features_errors[2] = std_features_base[2]
        std_features_errors[3] = std_features_base[3]
        std_features_errors[4] = std_features_base[4]
    if incl_corr: #take std of covariance to be product of std of features
        std_features_corr[0] = std_features_base[0]*std_features_base[1]
        std_features_corr[1] = std_features_base[0]*std_features_base[2]
        std_features_corr[2] = std_features_base[0]*std_features_base[3]
        std_features_corr[3] = std_features_base[0]*std_features_base[4]
        std_features_corr[4] = std_features_base[1]*std_features_base[2]
        std_features_corr[5] = std_features_base[1]*std_features_base[3]
        std_features_corr[6] = std_features_base[1]*std_features_base[4]
        std_features_corr[7] = std_features_base[2]*std_features_base[3]
        std_features_corr[8] = std_features_base[2]*std_features_base[4]
        std_features_corr[9] = std_features_base[3]*std_features_base[4]
    
    #store normalization parameters in file
    normfile = open(normfile_name, "w")
    for i in range(len(mean_features_base)):
        normfile.write(str(mean_features_base[i])+'\n')
        normfile.write(str(std_features_base[i])+'\n')
    if incl_vweight:
        for i in range(len(mean_features_vweight)):
            normfile.write(str(mean_features_vweight[i])+'\n')
            normfile.write(str(std_features_vweight[i])+'\n')
    if incl_errors:
        for i in range(len(mean_features_errors)):
            normfile.write(str(mean_features_errors[i])+'\n')
            normfile.write(str(std_features_errors[i])+'\n')
    if incl_corr:
        for i in range(len(mean_features_corr)):
            normfile.write(str(mean_features_corr[i])+'\n')
            normfile.write(str(std_features_corr[i])+'\n')
    if incl_hits:
        for i in range(len(mean_features_hits)):
            normfile.write(str(mean_features_hits[i])+'\n')
            normfile.write(str(std_features_hits[i])+'\n')
    normfile.close()

    #apply normalization from training data to all graph features
    print("Normalizing {} training graphs".format(len(train_graphs)))
    for graph in train_graphs:
        features_base = graph.ndata['features_base'].numpy()
        normed_features_base = np.divide(features_base-mean_features_base, std_features_base)
        graph.ndata['features_base'] = th.from_numpy(normed_features_base)
        if incl_vweight:
            features_vweight = graph.ndata['features_vweight'].numpy()
            normed_features_vweight = np.divide(features_vweight-mean_features_vweight, std_features_vweight)
            graph.ndata['features_vweight'] = th.from_numpy(normed_features_vweight) 
        if incl_errors:
            features_errors = graph.ndata['features_errors'].numpy()
            normed_features_errors = np.divide(features_errors-mean_features_errors, std_features_errors)
            graph.ndata['features_errors'] = th.from_numpy(normed_features_errors)
        if incl_corr:
            features_corr = graph.ndata['features_corr'].numpy()
            normed_features_corr = np.divide(features_corr-mean_features_corr, std_features_corr)
            graph.ndata['features_corr'] = th.from_numpy(normed_features_corr)
        if incl_hits:
            features_hits = graph.ndata['features_hits'].numpy()
            normed_features_hits = np.divide(features_hits-mean_features_hits, std_features_hits)
            graph.ndata['features_hits'] = th.from_numpy(normed_features_hits)
    dgl.save_graphs(train_outfile_name, train_graphs)

    val_graphs = dgl.load_graphs(val_infile_name)[0]
    print("Normalizing {} validation graphs".format(len(val_graphs)))
    for graph in val_graphs:
        features_base = graph.ndata['features_base'].numpy()
        normed_features_base = np.divide(features_base-mean_features_base, std_features_base)
        graph.ndata['features_base'] = th.from_numpy(normed_features_base)
        if incl_vweight:
            features_vweight = graph.ndata['features_vweight'].numpy()
            normed_features_vweight = np.divide(features_vweight-mean_features_vweight, std_features_vweight)
            graph.ndata['features_vweight'] = th.from_numpy(normed_features_vweight) 
        if incl_errors:
            features_errors = graph.ndata['features_errors'].numpy()
            normed_features_errors = np.divide(features_errors-mean_features_errors, std_features_errors)
            graph.ndata['features_errors'] = th.from_numpy(normed_features_errors)
        if incl_corr:
            features_corr = graph.ndata['features_corr'].numpy()
            normed_features_corr = np.divide(features_corr-mean_features_corr, std_features_corr)
            graph.ndata['features_corr'] = th.from_numpy(normed_features_corr)
        if incl_hits:
            features_hits = graph.ndata['features_hits'].numpy()
            normed_features_hits = np.divide(features_hits-mean_features_hits, std_features_hits)
            graph.ndata['features_hits'] = th.from_numpy(normed_features_hits)
    dgl.save_graphs(val_outfile_name, val_graphs)

    test_graphs = dgl.load_graphs(test_infile_name)[0]
    print("Normalizing {} testing graphs".format(len(test_graphs)))
    for graph in test_graphs:
        features_base = graph.ndata['features_base'].numpy()
        normed_features_base = np.divide(features_base-mean_features_base, std_features_base)
        graph.ndata['features_base'] = th.from_numpy(normed_features_base)
        if incl_vweight:
            features_vweight = graph.ndata['features_vweight'].numpy()
            normed_features_vweight = np.divide(features_vweight-mean_features_vweight, std_features_vweight)
            graph.ndata['features_vweight'] = th.from_numpy(normed_features_vweight) 
        if incl_errors:
            features_errors = graph.ndata['features_errors'].numpy()
            normed_features_errors = np.divide(features_errors-mean_features_errors, std_features_errors)
            graph.ndata['features_errors'] = th.from_numpy(normed_features_errors)
        if incl_corr:
            features_corr = graph.ndata['features_corr'].numpy()
            normed_features_corr = np.divide(features_corr-mean_features_corr, std_features_corr)
            graph.ndata['features_corr'] = th.from_numpy(normed_features_corr)
        if incl_hits:
            features_hits = graph.ndata['features_hits'].numpy()
            normed_features_hits = np.divide(features_hits-mean_features_hits, std_features_hits)
            graph.ndata['features_hits'] = th.from_numpy(normed_features_hits)
    dgl.save_graphs(test_outfile_name, test_graphs)


if __name__ == '__main__':
    main(sys.argv)
