import dgl
import torch as th
import os,sys,math,glob,ROOT
from ROOT import TH1D, TCanvas, gROOT, TProfile
import numpy as np
import argparse

import options
from plot_functions import *
from truth_functions import *

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

    b_pdgids = wd_bm + wd_bb #defined in truth_functions.py
    c_pdgids = wd_cm + wd_cb

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
    incl_errors = incl_corr = incl_hits = incl_vweight = False
    nnfeatures_base = train_graphs[0].ndata['features_base'].size()[1]
    nnfeatures = nnfeatures_base
    if 'features_vweight' in train_graphs[0].ndata.keys():
        nnfeatures_vweight = train_graphs[0].ndata['features_vweight'].size()[1]
        incl_vweight = True
        nnfeatures += nnfeatures_vweight
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

    hist_trk_pt_b = TH1D("trk_pt_b", "Track q/pT in dataset;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_btoc = TH1D("trk_pt_btoc", "Track q/pT in dataset;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_c = TH1D("trk_pt_c", "Track q/pT in dataset;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])

    hist_trk_theta_b = TH1D("trk_theta_b", "Track #theta in dataset;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_btoc = TH1D("trk_theta_btoc", "Track #theta in dataset;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_c = TH1D("trk_theta_c", "Track #theta in dataset;#theta;Normalized entries", 20, 0, math.pi)

    hist_trk_phi_b = TH1D("trk_phi_b", "Track #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc = TH1D("trk_phi_btoc", "Track #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c = TH1D("trk_phi_c", "Track #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    
    hist_trk_d0_b = TH1D("trk_d0_b", "Track d0 in dataset;d0 [mm];Normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_btoc = TH1D("trk_d0_btoc", "Track d0 in dataset;d0 [mm];Normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_c = TH1D("trk_d0_c", "Track d0 in dataset;d0 [mm];Normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    
    hist_trk_z0_b = TH1D("trk_z0_b", "Track z0 in dataset;z0 [mm];Normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_btoc = TH1D("trk_z0_btoc", "Track z0 in dataset;z0 [mm];Normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_c = TH1D("trk_z0_c", "Track z0 in dataset;z0 [mm];Normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    
    hist_jet_pt_b = TH1D("jet_pt_b", "Jet pT in dataset;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_pt_c = TH1D("jet_pt_c", "Jet pT in dataset;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_pt_l = TH1D("jet_pt_l", "Jet pT in dataset;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    
    hist_jet_eta_b = TH1D("jet_eta_b", "Jet #eta in dataset;#eta;Normalized entries", 20, jet_eta_bound[0], jet_eta_bound[1])
    hist_jet_eta_c = TH1D("jet_eta_c", "Jet #eta in dataset;#eta;Normalized entries", 20, jet_eta_bound[0], jet_eta_bound[1])
    hist_jet_eta_l = TH1D("jet_eta_l", "Jet #eta in dataset;#eta;Normalized entries", 20, jet_eta_bound[0], jet_eta_bound[1])
    
    hist_jet_phi_b = TH1D("jet_phi_b", "Jet #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_jet_phi_c = TH1D("jet_phi_c", "Jet #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_jet_phi_l = TH1D("jet_phi_l", "Jet #phi in dataset;#phi;Normalized entries", 20, -math.pi, math.pi)

    hist_acc_trk_pt_b = TH1D("acc_trk_pt_b", "Number of HF tracks in b-jets as a function of track pT;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_acc_trk_pt_c = TH1D("acc_trk_pt_c", "Number of HF tracks in c-jets as a function of track pT;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_rej_trk_pt_b = TH1D("rej_trk_pt_b", "Number of HF tracks in b-jets as a function of track pT;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_rej_trk_pt_c = TH1D("rej_trk_pt_c", "Number of HF tracks in c-jets as a function of track pT;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    
    hist_acc_trk_z0_b = TH1D("acc_trk_z0_b", "Number of HF tracks in b-jets as a function of track z0;z0 [mm];Entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_acc_trk_z0_c = TH1D("acc_trk_z0_c", "Number of HF tracks in c-jets as a function of track z0;z0 [mm];Entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_rej_trk_z0_b = TH1D("rej_trk_z0_b", "Number of HF tracks in b-jets as a function of track z0;z0 [mm];Entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_rej_trk_z0_c = TH1D("rej_trk_z0_c", "Number of HF tracks in c-jets as a function of track z0;z0 [mm];Entries", 20, track_z0_bound[0], track_z0_bound[1])

    hist_acc_trk_lxy_b = TH1D("acc_trk_lxy_b", "Number of HF tracks in b-jets as a function of vertex Lxy;Lxy [mm];Entries", 20, 0, 50)
    hist_acc_trk_lxy_c = TH1D("acc_trk_lxy_c", "Number of HF tracks in c-jets as a function of vertex Lxy;Lxy [mm];Entries", 20, 0, 50)
    hist_rej_trk_lxy_b = TH1D("rej_trk_lxy_b", "Number of HF tracks in b-jets as a function of vertex Lxy;Lxy [mm];Entries", 20, 0, 50)
    hist_rej_trk_lxy_c = TH1D("rej_trk_lxy_c", "Number of HF tracks in c-jets as a function of vertex Lxy;Lxy [mm];Entries", 20, 0, 50)

    hist_acc_trk_id_b = TH1D("acc_trk_id_b", "bH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))
    hist_acc_trk_id_c = TH1D("acc_trk_id_c", "prompt cH tracks as a function of cH ancestor PDG ID;PDG ID;Number of tracks",len(c_pdgids),0,len(c_pdgids))
    hist_acc_trk_id_btoc = TH1D("acc_trk_id_btoc", "bH->cH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))
    hist_rej_trk_id_b = TH1D("rej_trk_id_b", "bH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))
    hist_rej_trk_id_c = TH1D("rej_trk_id_c", "prompt cH tracks as a function of cH ancestor PDG ID;PDG ID;Number of tracks",len(c_pdgids),0,len(c_pdgids))
    hist_rej_trk_id_btoc = TH1D("rej_trk_id_btoc", "bH->cH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))

    prof_frac_reco_eta_b = TProfile("frac_reco_eta_b", "Fraction of jets with matched tracks and GNN reconstructible vertices;Jet #eta;Jet fraction",20,jet_eta_bound[0],jet_eta_bound[1])
    prof_frac_reco_eta_c = TProfile("frac_reco_eta_c", "Fraction of jets with matched tracks and GNN reconstructible vertices;Jet #eta;Jet fraction",20,jet_eta_bound[0],jet_eta_bound[1])

    hist_vertex_lxy_b = TH1D("vertex_lxy_b", "Vertex Lxy in dataset;Lxy [mm];Normalized entries", 20, 0, 50)
    hist_vertex_lxy_c = TH1D("vertex_lxy_c", "Vertex Lxy in dataset;Lxy [mm];Normalized entries", 20, 0, 50)

    bin_edges = np.linspace(-0.5,30.5,32)
    hist_no_trk_b = TH1D("no_trk_b", "Number of tracks per jet in dataset;Number of tracks;Normalized entries", 31, bin_edges)
    hist_no_trk_c = TH1D("no_trk_c", "Number of tracks per jet in dataset;Number of tracks;Normalized entries", 31, bin_edges)
    hist_no_trk_l = TH1D("no_trk_l", "Number of tracks per jet in dataset;Number of tracks;Normalized entries", 31, bin_edges)
    
    if incl_vweight:
        hist_acc_trk_vweight_hf = TH1D("acc_trk_vweight_hf", "Track to vertex association weight of HF tracks;Weight;Entries",20,0,1)
        hist_acc_trk_vweight_nohf = TH1D("acc_trk_vweight_nohf", "Track to vertex association weight of non HF tracks;Weight;Entries",20,0,1)
        hist_rej_trk_vweight_hf = TH1D("rej_trk_vweight_hf", "Track to vertex association weight of HF tracks;Weight;Entries",20,0,1)
        hist_rej_trk_vweight_nohf = TH1D("rej_trk_vweight_nohf", "Track to vertex association weight of non HF tracks;Weight;Entries",20,0,1) 

    if incl_errors:
        hist_trk_p_err_b = TH1D("trk_p_err_b", "Track 1/p error in dataset;1/p error [1/MeV];Normalized entries", 20, -0.0001, 0.0001)
        hist_trk_p_err_btoc = TH1D("trk_p_err_btoc", "Track 1/p error in dataset;1/p error [1/MeV];Normalized entries", 20, -0.0001, 0.0001)
        hist_trk_p_err_c = TH1D("trk_p_err_c", "Track 1/p in dataset;1/p error [1/MeV];Normalized entries", 20, -0.0001, 0.0001)
        
        hist_trk_theta_err_b = TH1D("trk_theta_err_b", "Track #theta error in dataset;#theta error;Normalized entries", 20, 0, 0.001)
        hist_trk_theta_err_btoc = TH1D("trk_theta_err_btoc", "Track #theta error in dataset;#theta error;Normalized entries", 20, 0, 0.001)
        hist_trk_theta_err_c = TH1D("trk_theta_err_c", "Track #theta error in dataset;#theta error;Normalized entries", 20, 0, 0.001)
        
        hist_trk_phi_err_b = TH1D("trk_phi_err_b", "Track #phi error in dataset;#phi error;Normalized entries", 20, 0, 0.01)
        hist_trk_phi_err_btoc = TH1D("trk_phi_err_btoc", "Track #phi error in dataset;#phi error;Normalized entries", 20, 0, 0.01)
        hist_trk_phi_err_c = TH1D("trk_phi_err_c", "Track #phi error in dataset;#phi error;Normalized entries", 20, 0, 0.01)
        
        hist_trk_d0_err_b = TH1D("trk_d0_err_b", "Track d0 error in dataset;d0 error [mm];Normalized entries", 20, 0, 1)
        hist_trk_d0_err_btoc = TH1D("trk_d0_err_btoc", "Track d0 error in dataset;d0 error [mm];Normalized entries", 20, 0, 1)
        hist_trk_d0_err_c = TH1D("trk_d0_err_c", "Track d0 error in dataset;d0 error [mm];Normalized entries", 20, 0, 1)
        
        hist_trk_z0_err_b = TH1D("trk_z0_err_b", "Track z0 error in dataset;z0 error [mm];Normalized entries", 20, 0, 0.5)
        hist_trk_z0_err_btoc = TH1D("trk_z0_err_btoc", "Track z0 error in dataset;z0 error [mm];Normalized entries", 20, 0, 0.5)
        hist_trk_z0_err_c = TH1D("trk_z0_err_c", "Track z0 error in dataset;z0 error [mm];Normalized entries", 20, 0, 0.5)

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

    #fill histograms using testing, training and validation files
    filename_array = [train_file_name, val_file_name, test_file_name]
    jet_classification = np.zeros((3,3))
    for filename in filename_array:
        graphs = dgl.load_graphs(filename)[0]
        for graph in graphs:
            passed_cuts = graph.ndata['passed_cuts'].numpy()
            features_base = graph.ndata['features_base'].numpy()
            if incl_errors: features_errors = graph.ndata['features_errors'].numpy()
            if incl_vweight: features_vweight = graph.ndata['features_vweight'].numpy()

            ancestor_pdgids = graph.ndata['node_info'].numpy()[:,1]
            second_ancestor_pdgids = graph.ndata['node_info'].numpy()[:,3]
            track_flavors = graph.ndata['node_info'].numpy()[:,4]
            jet_flavor = int(graph.ndata['graph_info'].numpy()[0,0])
            reco_jet_flavor = 0

            jet_pv = [graph.ndata['graph_info'].numpy()[0,1], graph.ndata['graph_info'].numpy()[0,2], graph.ndata['graph_info'].numpy()[0,3]]
            used_sv = [] #array containing secondary vertices that were already considered for relevant plots

            passed_tracks = 0 
            for i in range(len(features_base[:,0])):
                track_sv = [graph.ndata['node_info'].numpy()[i,5], graph.ndata['node_info'].numpy()[i,6], graph.ndata['node_info'].numpy()[i,7]]
                if passed_cuts[i] == 1:
                    passed_tracks += 1
                    if track_flavors[i] == 1:
                        hist_trk_pt_b.Fill(features_base[i,0])
                        hist_trk_theta_b.Fill(features_base[i,1])
                        hist_trk_phi_b.Fill(features_base[i,2])
                        hist_trk_d0_b.Fill(features_base[i,3])
                        hist_trk_z0_b.Fill(features_base[i,4])
                        hist_acc_trk_id_b.Fill(b_pdgids.index(abs(ancestor_pdgids[i])))
                        if track_sv not in used_sv:
                            hist_vertex_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                            used_sv.append(track_sv)
                        if incl_errors:
                            hist_trk_p_err_b.Fill(features_errors[i,0])
                            hist_trk_theta_err_b.Fill(features_errors[i,1])
                            hist_trk_phi_err_b.Fill(features_errors[i,2])
                            hist_trk_d0_err_b.Fill(features_errors[i,3])
                            hist_trk_z0_err_b.Fill(features_errors[i,4])
                        reco_jet_flavor = 1
                    elif track_flavors[i] == 3:
                        hist_trk_pt_btoc.Fill(features_base[i,0])
                        hist_trk_theta_btoc.Fill(features_base[i,1])
                        hist_trk_phi_btoc.Fill(features_base[i,2])
                        hist_trk_d0_btoc.Fill(features_base[i,3])
                        hist_trk_z0_btoc.Fill(features_base[i,4])
                        hist_acc_trk_id_btoc.Fill(b_pdgids.index(abs(second_ancestor_pdgids[i])))
                        if track_sv not in used_sv:
                            hist_vertex_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                            used_sv.append(track_sv)
                        if incl_errors:
                            hist_trk_p_err_btoc.Fill(features_errors[i,0])
                            hist_trk_theta_err_btoc.Fill(features_errors[i,1])
                            hist_trk_phi_err_btoc.Fill(features_errors[i,2])
                            hist_trk_d0_err_btoc.Fill(features_errors[i,3])
                            hist_trk_z0_err_btoc.Fill(features_errors[i,4])
                        reco_jet_flavor = 1
                    elif track_flavors[i] == 2:
                        hist_trk_pt_c.Fill(features_base[i,0])
                        hist_trk_theta_c.Fill(features_base[i,1])
                        hist_trk_phi_c.Fill(features_base[i,2])
                        hist_trk_d0_c.Fill(features_base[i,3])
                        hist_trk_z0_c.Fill(features_base[i,4])
                        hist_acc_trk_id_c.Fill(c_pdgids.index(abs(ancestor_pdgids[i])))
                        if track_sv not in used_sv:
                            hist_vertex_lxy_c.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                            used_sv.append(track_sv)
                        if incl_errors:
                            hist_trk_p_err_c.Fill(features_errors[i,0])
                            hist_trk_theta_err_c.Fill(features_errors[i,1])
                            hist_trk_phi_err_c.Fill(features_errors[i,2])
                            hist_trk_d0_err_c.Fill(features_errors[i,3])
                            hist_trk_z0_err_c.Fill(features_errors[i,4])
                        if not reco_jet_flavor: reco_jet_flavor = 2
                    
                    if incl_vweight:
                        if track_flavors[i] == 0 or track_flavors[i] == 4:
                            hist_acc_trk_vweight_nohf.Fill(features_vweight[i,0])
                        else:
                            hist_acc_trk_vweight_hf.Fill(features_vweight[i,0])

                    if jet_flavor == 1:
                        hist_acc_trk_pt_b.Fill(abs(1/features_base[i,0]))
                        hist_acc_trk_z0_b.Fill(features_base[i,4])
                        hist_acc_trk_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                    elif jet_flavor == 2:
                        hist_acc_trk_pt_c.Fill(abs(1/features_base[i,0]))
                        hist_acc_trk_z0_c.Fill(features_base[i,4])
                        hist_acc_trk_lxy_c.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))

                else:
                    if track_flavors[i] == 1:
                        hist_rej_trk_id_b.Fill(b_pdgids.index(abs(ancestor_pdgids[i])))
                    elif track_flavors[i] == 3:
                        hist_rej_trk_id_btoc.Fill(b_pdgids.index(abs(second_ancestor_pdgids[i])))
                    elif track_flavors[i] == 2:
                        hist_rej_trk_id_c.Fill(c_pdgids.index(abs(ancestor_pdgids[i])))
                    if incl_vweight:
                        if track_flavors[i] == 0 or track_flavors[i] == 4:
                            hist_rej_trk_vweight_nohf.Fill(features_vweight[i,0])
                        else:
                            hist_rej_trk_vweight_hf.Fill(features_vweight[i,0])
                    if jet_flavor == 1:
                        hist_rej_trk_pt_b.Fill(abs(1/features_base[i,0]))
                        hist_rej_trk_z0_b.Fill(features_base[i,4])
                        hist_rej_trk_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                    elif jet_flavor == 2:
                        hist_rej_trk_pt_c.Fill(abs(1/features_base[i,0]))
                        hist_rej_trk_z0_c.Fill(features_base[i,4])
                        hist_rej_trk_lxy_c.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))

            jet_classification[jet_flavor,reco_jet_flavor] += 1
            if jet_flavor == 1:
                hist_jet_pt_b.Fill(features_base[0,5])
                hist_jet_eta_b.Fill(features_base[0,6])
                hist_jet_phi_b.Fill(features_base[0,7])
                hist_no_trk_b.Fill(passed_tracks)
                if reco_jet_flavor == 1: reco_jet_flavor = 1
                elif reco_jet_flavor == 2: reco_jet_flavor = 0
                prof_frac_reco_eta_b.Fill(features_base[0,6], reco_jet_flavor)
            elif jet_flavor == 2:
                hist_jet_pt_c.Fill(features_base[0,5])
                hist_jet_eta_c.Fill(features_base[0,6])
                hist_jet_phi_c.Fill(features_base[0,7])
                hist_no_trk_c.Fill(passed_tracks)
                if reco_jet_flavor == 1: reco_jet_flavor = 0
                elif reco_jet_flavor == 2: reco_jet_flavor = 1
                prof_frac_reco_eta_c.Fill(features_base[0,6], reco_jet_flavor)
            else:
                hist_jet_pt_l.Fill(features_base[0,5])
                hist_jet_eta_l.Fill(features_base[0,6])
                hist_jet_phi_l.Fill(features_base[0,7])
                hist_no_trk_l.Fill(passed_tracks)

    plot_confusion_matrix(jet_classification, ["l","b","c"], ["l","b","c"], "", ["Target jet classification", "Jet flavor label"], data_path+data_name+"_jet_cm.png")
    plot_hist([hist_trk_pt_b, hist_trk_pt_btoc, hist_trk_pt_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_pt.png", "HIST", scaling=[mean_features[0],std_features[0]])
    plot_hist([hist_trk_theta_b, hist_trk_theta_btoc, hist_trk_theta_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_theta.png", "HIST", scaling=[mean_features[1],std_features[1]])
    plot_hist([hist_trk_phi_b, hist_trk_phi_btoc, hist_trk_phi_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_phi.png", "HIST", scaling=[mean_features[2],std_features[2]])
    plot_hist([hist_trk_d0_b, hist_trk_d0_btoc, hist_trk_d0_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_d0.png", "HIST", scaling=[mean_features[3],std_features[3]])
    plot_hist([hist_trk_z0_b, hist_trk_z0_btoc, hist_trk_z0_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_z0.png", "HIST", scaling=[mean_features[4],std_features[4]])
    plot_hist([hist_jet_pt_b, hist_jet_pt_c, hist_jet_pt_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_jet_pt.png", "HIST", scaling=[mean_features[5],std_features[5]])
    plot_hist([hist_jet_eta_b, hist_jet_eta_c, hist_jet_eta_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_jet_eta.png", "HIST", scaling=[mean_features[6],std_features[6]])
    plot_hist([hist_jet_phi_b, hist_jet_phi_c, hist_jet_phi_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_jet_phi.png", "HIST", scaling=[mean_features[7],std_features[7]])
    plot_hist([hist_no_trk_b, hist_no_trk_c, hist_no_trk_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_no_trk.png", "HIST")
    plot_hist([hist_vertex_lxy_b, hist_vertex_lxy_c], ["b", "c"], True, True, True, data_path+data_name+"_lxy.png", "HIST")
    plot_profile([prof_frac_reco_eta_b, prof_frac_reco_eta_c], ["b", "c"], [0.0, 1.4], False, data_path+data_name+"_frac_reco_eta.png")

    if incl_errors:
        plot_hist([hist_trk_p_err_b, hist_trk_p_err_btoc, hist_trk_p_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_p_err.png", "HIST", scaling=[mean_features[8],std_features[8]])
        plot_hist([hist_trk_theta_err_b, hist_trk_theta_err_btoc, hist_trk_theta_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_theta_err.png", "HIST", scaling=[mean_features[9],std_features[9]])
        plot_hist([hist_trk_phi_err_b, hist_trk_phi_err_btoc, hist_trk_phi_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_phi_err.png", "HIST", scaling=[mean_features[10],std_features[10]])
        plot_hist([hist_trk_d0_err_b, hist_trk_d0_err_btoc, hist_trk_d0_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_d0_err.png", "HIST", scaling=[mean_features[11],std_features[11]])
        plot_hist([hist_trk_z0_err_b, hist_trk_z0_err_btoc, hist_trk_z0_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_z0_err.png", "HIST", scaling=[mean_features[12],std_features[12]])

    if incl_vweight:
        plot_hist([hist_acc_trk_vweight_hf, hist_rej_trk_vweight_hf], ["passsed", "cut"], False, True, True, data_path+data_name+"_cut_trk_vweight_hf.png", "HIST")
        plot_hist([hist_acc_trk_vweight_nohf, hist_rej_trk_vweight_nohf], ["passsed", "cut"], False, True, True, data_path+data_name+"_cut_trk_vweight_nohf.png", "HIST")

    plot_hist([hist_acc_trk_pt_b, hist_rej_trk_pt_b], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_pt_b.png", "HIST")
    plot_hist([hist_acc_trk_pt_c, hist_rej_trk_pt_c], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_pt_c.png", "HIST")
    plot_hist([hist_acc_trk_z0_b, hist_rej_trk_z0_b], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_z0_b.png", "HIST")
    plot_hist([hist_acc_trk_z0_c, hist_rej_trk_z0_c], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_z0_c.png", "HIST")
    plot_hist([hist_acc_trk_lxy_b, hist_rej_trk_lxy_b], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_lxy_b.png", "HIST")
    plot_hist([hist_acc_trk_lxy_c, hist_rej_trk_lxy_c], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_lxy_c.png", "HIST")
    plot_bar([hist_acc_trk_id_b, hist_rej_trk_id_b], b_pdgids, ["passed", "cut"], False, True, data_path+data_name+"_trk_ancid_b.png", "HIST")
    plot_bar([hist_acc_trk_id_c, hist_rej_trk_id_c], c_pdgids, ["passed", "cut"], False, True, data_path+data_name+"_trk_ancid_c.png", "HIST")
    plot_bar([hist_acc_trk_id_btoc, hist_rej_trk_id_btoc], b_pdgids, ["passed", "cut"], False, True, data_path+data_name+"_trk_ancid_btoc.png", "HIST")


if __name__ == '__main__':
    main(sys.argv)
