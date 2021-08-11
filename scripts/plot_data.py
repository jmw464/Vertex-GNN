import dgl
import torch as th
import os,sys,math,glob,ROOT
from ROOT import TH1D, TCanvas, gROOT
import numpy as np
import argparse

import options
from plot_functions import *

#set ATLAS style for plots
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasStyle.C")
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasLabels.C")
from ROOT import SetAtlasStyle


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")
    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name

    #import options from option file
    atlasstyle = options.use_atlas_style
    track_pt_bound = options.track_pt_bound
    track_d0_bound = options.track_d0_bound
    track_z0_bound = options.track_z0_bound
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound

    if atlasstyle: SetAtlasStyle()

    train_file_name = data_path+data_name+"_train.bin"
    val_file_name = data_path+data_name+"_val.bin"
    test_file_name = data_path+data_name+"_test.bin"
    normfile_name = data_path+data_name+"_norm"

    train_graphs = dgl.load_graphs(train_file_name)[0]

    #calculate number of features in graphs
    incl_errors = incl_corr = incl_hits = False
    nnfeatures_base = train_graphs[0].ndata['features_base'].size()[1]
    nnfeatures = nnfeatures_base
    if 'features_errors' in train_graphs[0].ndata.keys():
        nnfeatures_errors = train_graphs[0].ndata['features_errors'].size()[1]
        incl_errors = True
        nnfeatures += nnfeatures_errors
    if 'features_hits' in train_graphs[0].ndata.keys():
        nnfeatures_hits = train_graphs[0].ndata['features_hits'].size()[1]
        incl_hits = True
        nnfeatures += nnfeatures_hits
    if 'features_corr' in train_graphs[0].ndata.keys():
        nnfeatures_corr = train_graphs[0].ndata['features_corr'].size()[1]
        incl_corr = True
        nnfeatures += nnfeatures_corr

    hist_trk_pt_train = TH1D("trk_pt_train", "Track q/pT in dataset;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_val = TH1D("trk_pt_val", "Track q/pT in dataset;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_test = TH1D("trk_pt_test", "Track q/pT in dataset;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])

    hist_trk_theta_train = TH1D("trk_theta_train", "Track #theta in dataset;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_val = TH1D("trk_theta_val", "Track #theta in dataset;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_test = TH1D("trk_theta_test", "Track #theta in dataset;#theta;Normalized entries", 20, 0, math.pi)

    hist_trk_phi_train = TH1D("trk_phi_train", "Track #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_val = TH1D("trk_phi_val", "Track #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_test = TH1D("trk_phi_test", "Track #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    
    hist_trk_d0_train = TH1D("trk_d0_train", "Track d0 in dataset;d0 [cm];Normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_val = TH1D("trk_d0_val", "Track d0 in dataset;d0 [cm];Normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_test = TH1D("trk_d0_test", "Track d0 in dataset;d0 [cm];Normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    
    hist_trk_z0_train = TH1D("trk_z0_train", "Track z0 in dataset;z0 [cm];Normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_val = TH1D("trk_z0_val", "Track z0 in dataset;z0 [cm];Normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_test = TH1D("trk_z0_test", "Track z0 in dataset;z0 [cm];Normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    
    hist_jet_pt_train = TH1D("jet_pt_train", "Jet pT in dataset;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_pt_val = TH1D("jet_pt_val", "Jet pT in dataset;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_pt_test = TH1D("jet_pt_test", "Jet pT in dataset;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    
    hist_jet_eta_train = TH1D("jet_eta_train", "Jet #eta in dataset;#eta;Normalized entries", 20, jet_eta_bound[0], jet_eta_bound[1])
    hist_jet_eta_val = TH1D("jet_eta_val", "Jet #eta in dataset;#eta;Normalized entries", 20, jet_eta_bound[0], jet_eta_bound[1])
    hist_jet_eta_test = TH1D("jet_eta_test", "Jet #eta in dataset;#eta;Normalized entries", 20, jet_eta_bound[0], jet_eta_bound[1])
    
    hist_jet_phi_train = TH1D("jet_phi_train", "Jet #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_jet_phi_val = TH1D("jet_phi_val", "Jet #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_jet_phi_test = TH1D("jet_phi_test", "Jet #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    
    bin_edges = np.linspace(-0.5,10.5,12)
    hist_no_trk_train = TH1D("no_trk_train", "Number of tracks per jet in dataset;Number of tracks;Normalized entries", 11, bin_edges)
    hist_no_trk_val = TH1D("no_trk_val", "Number of tracks per jet in dataset;Number of tracks;Normalized entries", 11, bin_edges)
    hist_no_trk_test = TH1D("no_trk_test", "Number of tracks per jet in dataset;Number of tracks;Normalized entries", 11, bin_edges)
     
    if incl_errors:
        hist_trk_p_err_train = TH1D("trk_p_err_train", "Track 1/p error in dataset;1/p error [1/MeV];Normalized entries", 20, -0.0001, 0.0001)
        hist_trk_p_err_val = TH1D("trk_p_err_val", "Track 1/p error in dataset;1/p error [1/MeV];Normalized entries", 20, -0.0001, 0.0001)
        hist_trk_p_err_test = TH1D("trk_p_err_test", "Track 1/p in dataset;1/p error [1/MeV];Normalized entries", 20, -0.0001, 0.0001)
        
        hist_trk_theta_err_train = TH1D("trk_theta_err_train", "Track #theta error in dataset;#theta error;Normalized entries", 20, 0, 0.01)
        hist_trk_theta_err_val = TH1D("trk_theta_err_val", "Track #theta error in dataset;#theta error;Normalized entries", 20, 0, 0.01)
        hist_trk_theta_err_test = TH1D("trk_theta_err_test", "Track #theta error in dataset;#theta error;Normalized entries", 20, 0, 0.01)
        
        hist_trk_phi_err_train = TH1D("trk_phi_err_train", "Track #phi error in dataset;#phi error;Normalized entries", 20, 0, 0.01)
        hist_trk_phi_err_val = TH1D("trk_phi_err_val", "Track #phi error in dataset;#phi error;Normalized entries", 20, 0, 0.01)
        hist_trk_phi_err_test = TH1D("trk_phi_err_test", "Track #phi error in dataset;#phi error;Normalized entries", 20, 0, 0.01)
        
        hist_trk_d0_err_train = TH1D("trk_d0_err_train", "Track d0 error in dataset;d0 error [cm];Normalized entries", 20, 0, 5)
        hist_trk_d0_err_val = TH1D("trk_d0_err_val", "Track d0 error in dataset;d0 error [cm];Normalized entries", 20, 0, 5)
        hist_trk_d0_err_test = TH1D("trk_d0_err_test", "Track d0 error in dataset;d0 error [cm];Normalized entries", 20, 0, 5)
        
        hist_trk_z0_err_train = TH1D("trk_z0_err_train", "Track z0 error in dataset;z0 error [cm];Normalized entries", 20, 0, 5)
        hist_trk_z0_err_val = TH1D("trk_z0_err_val", "Track z0 error in dataset;z0 error [cm];Normalized entries", 20, 0, 5)
        hist_trk_z0_err_test = TH1D("trk_z0_err_test", "Track z0 error in dataset;z0 error [cm];Normalized entries", 20, 0, 5)

    #read in normalization constants for features
    if os.path.isfile(normfile_name):
        normfile = open(normfile_name, "r")
        mean_features = np.zeros(nnfeatures)
        std_features = np.zeros(nnfeatures) 
        
        counter = 0
        for line in normfile:
            if int(counter%2) == 0:
                mean_features[int(counter/2)] = float(line)
            else:
                std_features[int((counter-1)/2)] = float(line)
            counter += 1

    #fill histograms
    for graph in train_graphs:
        features_base = graph.ndata['features_base'].numpy()
        for i in range(len(features_base[:,0])):
            hist_trk_pt_train.Fill(features_base[i,0])
            hist_trk_theta_train.Fill(features_base[i,1])
            hist_trk_phi_train.Fill(features_base[i,2])
            hist_trk_d0_train.Fill(features_base[i,3])
            hist_trk_z0_train.Fill(features_base[i,4])
        hist_jet_pt_train.Fill(features_base[0,5])
        hist_jet_eta_train.Fill(features_base[0,6])
        hist_jet_phi_train.Fill(features_base[0,7])
        hist_no_trk_train.Fill(graph.num_nodes())
        if incl_errors:
            features_errors = graph.ndata['features_errors'].numpy()
            for i in range(len(features_errors[:,0])):
                hist_trk_p_err_train.Fill(features_errors[i,0])
                hist_trk_theta_err_train.Fill(features_errors[i,1])
                hist_trk_phi_err_train.Fill(features_errors[i,2])
                hist_trk_d0_err_train.Fill(features_errors[i,3])
                hist_trk_z0_err_train.Fill(features_errors[i,4])

    val_graphs = dgl.load_graphs(val_file_name)[0]
    for graph in val_graphs:
        features_base = graph.ndata['features_base'].numpy()
        for i in range(len(features_base[:,0])):
            hist_trk_pt_val.Fill(features_base[i,0])
            hist_trk_theta_val.Fill(features_base[i,1])
            hist_trk_phi_val.Fill(features_base[i,2])
            hist_trk_d0_val.Fill(features_base[i,3])
            hist_trk_z0_val.Fill(features_base[i,4])
        hist_jet_pt_val.Fill(features_base[0,5])
        hist_jet_eta_val.Fill(features_base[0,6])
        hist_jet_phi_val.Fill(features_base[0,7])
        hist_no_trk_val.Fill(graph.num_nodes())
        if incl_errors:
            features_errors = graph.ndata['features_errors'].numpy()
            for i in range(len(features_errors[:,0])):
                hist_trk_p_err_val.Fill(features_errors[i,0])
                hist_trk_theta_err_val.Fill(features_errors[i,1])
                hist_trk_phi_err_val.Fill(features_errors[i,2])
                hist_trk_d0_err_val.Fill(features_errors[i,3])
                hist_trk_z0_err_val.Fill(features_errors[i,4])

    test_graphs = dgl.load_graphs(test_file_name)[0]
    for graph in test_graphs:
        features_base = graph.ndata['features_base'].numpy()
        for i in range(len(features_base[:,0])):
            hist_trk_pt_test.Fill(features_base[i,0])
            hist_trk_theta_test.Fill(features_base[i,1])
            hist_trk_phi_test.Fill(features_base[i,2])
            hist_trk_d0_test.Fill(features_base[i,3])
            hist_trk_z0_test.Fill(features_base[i,4])
        hist_jet_pt_test.Fill(features_base[0,5])
        hist_jet_eta_test.Fill(features_base[0,6])
        hist_jet_phi_test.Fill(features_base[0,7])
        hist_no_trk_test.Fill(graph.num_nodes())
        if incl_errors:
            features_errors = graph.ndata['features_errors'].numpy()
            for i in range(len(features_errors[:,0])):
                hist_trk_p_err_test.Fill(features_errors[i,0])
                hist_trk_theta_err_test.Fill(features_errors[i,1])
                hist_trk_phi_err_test.Fill(features_errors[i,2])
                hist_trk_d0_err_test.Fill(features_errors[i,3])
                hist_trk_z0_err_test.Fill(features_errors[i,4])

    plot_hist([hist_trk_pt_test, hist_trk_pt_val, hist_trk_pt_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_pt.png", "HIST", scaling=[mean_features[0],std_features[0]])
    plot_hist([hist_trk_theta_test, hist_trk_theta_val, hist_trk_theta_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_theta.png", "HIST", scaling=[mean_features[1],std_features[1]])
    plot_hist([hist_trk_phi_test, hist_trk_phi_val, hist_trk_phi_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_phi.png", "HIST", scaling=[mean_features[2],std_features[2]])
    plot_hist([hist_trk_d0_test, hist_trk_d0_val, hist_trk_d0_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_d0.png", "HIST", scaling=[mean_features[3],std_features[3]])
    plot_hist([hist_trk_z0_test, hist_trk_z0_val, hist_trk_z0_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_z0.png", "HIST", scaling=[mean_features[4],std_features[4]])
    plot_hist([hist_jet_pt_test, hist_jet_pt_val, hist_jet_pt_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_jet_pt.png", "HIST", scaling=[mean_features[5],std_features[5]])
    plot_hist([hist_jet_eta_test, hist_jet_eta_val, hist_jet_eta_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_jet_eta.png", "HIST", scaling=[mean_features[6],std_features[6]])
    plot_hist([hist_jet_phi_test, hist_jet_phi_val, hist_jet_phi_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_jet_phi.png", "HIST", scaling=[mean_features[7],std_features[7]])
    plot_hist([hist_no_trk_test, hist_no_trk_val, hist_no_trk_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_no_trk.png", "HIST")
    if incl_errors:
        plot_hist([hist_trk_p_err_test, hist_trk_p_err_val, hist_trk_p_err_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_p_err.png", "HIST", scaling=[mean_features[8],std_features[8]])
        plot_hist([hist_trk_theta_err_test, hist_trk_theta_err_val, hist_trk_theta_err_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_theta_err.png", "HIST", scaling=[mean_features[9],std_features[9]])
        plot_hist([hist_trk_phi_err_test, hist_trk_phi_err_val, hist_trk_phi_err_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_phi_err.png", "HIST", scaling=[mean_features[10],std_features[10]])
        plot_hist([hist_trk_d0_err_test, hist_trk_d0_err_val, hist_trk_d0_err_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_d0_err.png", "HIST", scaling=[mean_features[11],std_features[11]])
        plot_hist([hist_trk_z0_err_test, hist_trk_z0_err_val, hist_trk_z0_err_train], ["testing", "validation", "training"], True, False, True, data_path+data_name+"_trk_z0_err.png", "HIST", scaling=[mean_features[12],std_features[12]])


if __name__ == '__main__':
    main(sys.argv)
