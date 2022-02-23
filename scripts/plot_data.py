#!/usr/bin/env python

######################################## plot_data.py ########################################
# PURPOSE: make plots of input dataset
# EDIT TO: add more plots, update graph structure (if modified in create_graphs)
# -------------------------------------------Summary------------------------------------------
# This script generates plots from the given input dataset. It is run after combine_graphs,
# which means it takes in separate test/train/val files. The plots this script creates show
# selected track/jet feature distributions as well as cuts made on track-level as a function of
# various variables. Plots generally also reflect the given track label (b, c or btoc), jet
# flavor label (b, c or l) or vertex label (b or c). Note that track and vertex labels are
# calculated based on HF ancestors (in process_ntuple) while jet flavor labels are taken
# directly from the ntuple. As such these are not always consistent (such as in the case where
# all HF tracks within a jet are cut/misreconstructed before this stage, thus making the vertex
# invisible). Because of this, is why this script also generates a 3x3 confusion matrix
# comparing the two label types and a plot showing what fraction of total b and c jets have
# "GNN reconstructible" vertices.
###############################################################################################


import dgl
import torch as th
import os,sys,math,ROOT
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
    track_pt_err_bound = options.track_pt_err_bound
    track_d0_bound = options.track_d0_bound
    track_d0_err_bound = options.track_d0_err_bound
    track_z0_bound = options.track_z0_bound
    track_z0_err_bound = options.track_z0_err_bound
    track_phi_err_bound = options.track_phi_err_bound
    track_theta_err_bound = options.track_theta_err_bound
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound
    ntrk_bound = options.ntrk_bound
    lxy_bound = options.lxy_bound

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

    #histograms of basic track features by track label
    hist_trk_pt_b = TH1D("trk_pt_b", "Track q/pT in dataset for different track labels;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_btoc = TH1D("trk_pt_btoc", "Track q/pT in dataset for different track labels;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_c = TH1D("trk_pt_c", "Track q/pT in dataset for different track labels;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_pt_nohf = TH1D("trk_pt_nohf", "Track q/pT in dataset for different track labels;q/pT [1/GeV];Normalized entries", 20, -1/track_pt_bound[0], 1/track_pt_bound[0])
    hist_trk_theta_b = TH1D("trk_theta_b", "Track #theta in dataset for different track labels;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_btoc = TH1D("trk_theta_btoc", "Track #theta in dataset for different track labels;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_c = TH1D("trk_theta_c", "Track #theta in dataset for different track labels;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_nohf = TH1D("trk_theta_nohf", "Track #theta in dataset for different track labels;#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_phi_b = TH1D("trk_phi_b", "Track #phi in dataset for different track labels;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc = TH1D("trk_phi_btoc", "Track #phi in dataset for different track labels;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c = TH1D("trk_phi_c", "Track #phi in dataset for different track labels;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_nohf = TH1D("trk_phi_nohf", "Track #phi in dataset for different track labels;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_d0_b = TH1D("trk_d0_b", "Track d0 in dataset for different track labels;d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_btoc = TH1D("trk_d0_btoc", "Track d0 in dataset for different track labels;d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_c = TH1D("trk_d0_c", "Track d0 in dataset for different track labels;d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_nohf = TH1D("trk_d0_nohf", "Track d0 in dataset for different track labels;d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_z0_b = TH1D("trk_z0_b", "Track z0 in dataset for different track labels;z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_btoc = TH1D("trk_z0_btoc", "Track z0 in dataset for different track labels;z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_c = TH1D("trk_z0_c", "Track z0 in dataset for different track labels;z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_nohf = TH1D("trk_z0_nohf", "Track z0 in dataset for different track labels;z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)

    #histograms of basic jet features by jet flavor
    bin_edges = np.linspace(-0.5,ntrk_bound+0.5,ntrk_bound+2)
    hist_no_trk_b = TH1D("no_trk_b", "Number of tracks per jet in dataset for different jet flavors;Number of tracks;Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_c = TH1D("no_trk_c", "Number of tracks per jet in dataset for different jet flavors;Number of tracks;Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_l = TH1D("no_trk_l", "Number of tracks per jet in dataset for different jet flavors;Number of tracks;Normalized entries", ntrk_bound+1, bin_edges)
    hist_jet_pt_b = TH1D("jet_pt_b", "Jet pT in dataset for different jet flavors;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_pt_c = TH1D("jet_pt_c", "Jet pT in dataset for different jet flavors;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_pt_l = TH1D("jet_pt_l", "Jet pT in dataset for different jet flavors;pT [GeV];Normalized entries", 20, jet_pt_bound[0], jet_pt_bound[1])
    hist_jet_eta_b = TH1D("jet_eta_b", "Jet #eta in dataset for different jet flavors;#eta;Normalized entries", 20, -jet_eta_bound, jet_eta_bound)
    hist_jet_eta_c = TH1D("jet_eta_c", "Jet #eta in dataset for different jet flavors;#eta;Normalized entries", 20, -jet_eta_bound, jet_eta_bound)
    hist_jet_eta_l = TH1D("jet_eta_l", "Jet #eta in dataset for different jet flavors;#eta;Normalized entries", 20, -jet_eta_bound, jet_eta_bound)
    hist_jet_phi_b = TH1D("jet_phi_b", "Jet #phi in dataset for different jet flavors;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_jet_phi_c = TH1D("jet_phi_c", "Jet #phi in dataset for different jet flavors;#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_jet_phi_l = TH1D("jet_phi_l", "Jet #phi in dataset for different jet flavors;#phi;Normalized entries", 20, -math.pi, math.pi)

    #histograms of track error features by track label
    if incl_errors:
        hist_trk_p_err_b = TH1D("trk_p_err_b", "Track 1/p error in dataset for different track labels;1/p error [1/GeV];Normalized entries", 20, 0, 1./track_pt_err_bound[0])
        hist_trk_p_err_btoc = TH1D("trk_p_err_btoc", "Track 1/p error in dataset for different track labels;1/p error [1/GeV];Normalized entries", 20, 0, 1./track_pt_err_bound[0])
        hist_trk_p_err_c = TH1D("trk_p_err_c", "Track 1/p in dataset for different track labels;1/p error [1/GeV];Normalized entries", 20, 0, 1./track_pt_err_bound[0])
        hist_trk_p_err_nohf = TH1D("trk_p_err_nohf", "Track 1/p in dataset for different track labels;1/p error [1/GeV];Normalized entries", 20, 0, 1./track_pt_err_bound[0])        
        hist_trk_theta_err_b = TH1D("trk_theta_err_b", "Track #theta error in dataset for different track labels;#theta error;Normalized entries", 20, 0, track_theta_err_bound)
        hist_trk_theta_err_btoc = TH1D("trk_theta_err_btoc", "Track #theta error in dataset for different track labels;#theta error;Normalized entries", 20, 0, track_theta_err_bound)
        hist_trk_theta_err_c = TH1D("trk_theta_err_c", "Track #theta error in dataset for different track labels;#theta error;Normalized entries", 20, 0, track_theta_err_bound)
        hist_trk_theta_err_nohf = TH1D("trk_theta_err_nohf", "Track #theta error in dataset for different track labels;#theta error;Normalized entries", 20, 0, track_theta_err_bound)
        hist_trk_phi_err_b = TH1D("trk_phi_err_b", "Track #phi error in dataset for different track labels;#phi error;Normalized entries", 20, 0, track_phi_err_bound)
        hist_trk_phi_err_btoc = TH1D("trk_phi_err_btoc", "Track #phi error in dataset for different track labels;#phi error;Normalized entries", 20, 0, track_phi_err_bound)
        hist_trk_phi_err_c = TH1D("trk_phi_err_c", "Track #phi error in dataset for different track labels;#phi error;Normalized entries", 20, 0, track_phi_err_bound)
        hist_trk_phi_err_nohf = TH1D("trk_phi_err_nohf", "Track #phi error in dataset for different track labels;#phi error;Normalized entries", 20, 0, track_phi_err_bound)
        hist_trk_d0_err_b = TH1D("trk_d0_err_b", "Track d0 error in dataset for different track labels;d0 error [mm];Normalized entries", 20, 0, track_d0_err_bound)
        hist_trk_d0_err_btoc = TH1D("trk_d0_err_btoc", "Track d0 error in dataset for different track labels;d0 error [mm];Normalized entries", 20, 0, track_d0_err_bound)
        hist_trk_d0_err_c = TH1D("trk_d0_err_c", "Track d0 error in dataset for different track labels;d0 error [mm];Normalized entries", 20, 0, track_d0_err_bound)
        hist_trk_d0_err_nohf = TH1D("trk_d0_err_nohf", "Track d0 error in dataset for different track labels;d0 error [mm];Normalized entries", 20, 0, track_d0_err_bound)
        hist_trk_z0_err_b = TH1D("trk_z0_err_b", "Track z0 error in dataset for different track labels;z0 error [mm];Normalized entries", 20, 0, track_z0_err_bound)
        hist_trk_z0_err_btoc = TH1D("trk_z0_err_btoc", "Track z0 error in dataset for different track labels;z0 error [mm];Normalized entries", 20, 0, track_z0_err_bound)
        hist_trk_z0_err_c = TH1D("trk_z0_err_c", "Track z0 error in dataset for different track labels;z0 error [mm];Normalized entries", 20, 0, track_z0_err_bound)
        hist_trk_z0_err_nohf = TH1D("trk_z0_err_nohf", "Track z0 error in dataset for different track labels;z0 error [mm];Normalized entries", 20, 0, track_z0_err_bound)

    #histograms of Lxy by vertex label (same as track label except btoc -> b)
    hist_vertex_lxy_b = TH1D("vertex_lxy_b", "Vertex Lxy in dataset for different vertex labels;Lxy [mm];Normalized entries", 20, 0, lxy_bound)
    hist_vertex_lxy_c = TH1D("vertex_lxy_c", "Vertex Lxy in dataset for different vertex labels;Lxy [mm];Normalized entries", 20, 0, lxy_bound)

    #histograms of track cuts as a function of basic track features
    hist_acc_trk_pt_hf = TH1D("acc_trk_pt_hf", "Track pT of passed/cut HF tracks;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_acc_trk_pt_nohf = TH1D("acc_trk_pt_nohf", "Track pT of passed/cut non HF tracks;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_rej_trk_pt_hf = TH1D("rej_trk_pt_hf", "Track pT of passed/cut HF tracks;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_rej_trk_pt_nohf = TH1D("rej_trk_pt_nohf", "Track pT of passed/cut non HF tracks;pT [GeV];Entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_acc_trk_z0_hf = TH1D("acc_trk_z0_hf", "Track z0 of passed/cut HF tracks;z0 [mm];Entries", 20, -track_z0_bound, track_z0_bound)
    hist_acc_trk_z0_nohf = TH1D("acc_trk_z0_nohf", "Track z0 of passed/cut non HF tracks;z0 [mm];Entries", 20, -track_z0_bound, track_z0_bound)
    hist_rej_trk_z0_hf = TH1D("rej_trk_z0_hf", "Track z0 of passed/cut HF tracks;z0 [mm];Entries", 20, -track_z0_bound, track_z0_bound)
    hist_rej_trk_z0_nohf = TH1D("rej_trk_z0_nohf", "Track z0 of passed/cut non HF tracks;z0 [mm];Entries", 20, -track_z0_bound, track_z0_bound)

    #histograms of track cuts as a function of vertex weight feature
    if incl_vweight:
        hist_acc_trk_vweight_hf = TH1D("acc_trk_vweight_hf", "Track to vertex association weight of passed/cut HF tracks;Weight;Entries",20,0,1)
        hist_acc_trk_vweight_nohf = TH1D("acc_trk_vweight_nohf", "Track to vertex association weight of passed/cut non HF tracks;Weight;Entries",20,0,1)
        hist_rej_trk_vweight_hf = TH1D("rej_trk_vweight_hf", "Track to vertex association weight of passed/cut HF tracks;Weight;Entries",20,0,1)
        hist_rej_trk_vweight_nohf = TH1D("rej_trk_vweight_nohf", "Track to vertex association weight of passed/cut non HF tracks;Weight;Entries",20,0,1) 

    #histograms of track cuts as a function of Lxy
    hist_acc_trk_lxy_b = TH1D("acc_trk_lxy_b", "Vertex Lxy of passed/cut tracks in b-vertices;Lxy [mm];Entries", 20, 0, lxy_bound)
    hist_acc_trk_lxy_c = TH1D("acc_trk_lxy_c", "Vertex Lxy of passed/cut tracks in c-vertices;Lxy [mm];Entries", 20, 0, lxy_bound)
    hist_rej_trk_lxy_b = TH1D("rej_trk_lxy_b", "Vertex Lxy of passed/cut tracks in b-vertices;Lxy [mm];Entries", 20, 0, lxy_bound)
    hist_rej_trk_lxy_c = TH1D("rej_trk_lxy_c", "Vertex Lxy of passed/cut tracks in c-vertices;Lxy [mm];Entries", 20, 0, lxy_bound)

    #histograms of track cuts as a function of track origin PDGID
    hist_acc_trk_id_b = TH1D("acc_trk_id_b", "bH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))
    hist_acc_trk_id_c = TH1D("acc_trk_id_c", "prompt cH tracks as a function of cH ancestor PDG ID;PDG ID;Number of tracks",len(c_pdgids),0,len(c_pdgids))
    hist_acc_trk_id_btoc = TH1D("acc_trk_id_btoc", "bH->cH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))
    hist_rej_trk_id_b = TH1D("rej_trk_id_b", "bH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))
    hist_rej_trk_id_c = TH1D("rej_trk_id_c", "prompt cH tracks as a function of cH ancestor PDG ID;PDG ID;Number of tracks",len(c_pdgids),0,len(c_pdgids))
    hist_rej_trk_id_btoc = TH1D("rej_trk_id_btoc", "bH->cH tracks as a function of bH ancestor PDG ID;PDG ID;Number of tracks",len(b_pdgids),0,len(b_pdgids))

    #profiles of GNN reconstructible vertices as a function of basic jet features
    prof_frac_reco_pt_b = TProfile("frac_reco_pt_b", "Fraction of jets with matched tracks and GNN reconstructible vertices;Jet pT;Jet fraction",20,jet_pt_bound[0],jet_pt_bound[1])
    prof_frac_reco_pt_c = TProfile("frac_reco_pt_c", "Fraction of jets with matched tracks and GNN reconstructible vertices;Jet pT;Jet fraction",20,jet_pt_bound[0],jet_pt_bound[1])
    prof_frac_reco_eta_b = TProfile("frac_reco_eta_b", "Fraction of jets with matched tracks and GNN reconstructible vertices;Jet #eta;Jet fraction",20,-jet_eta_bound,jet_eta_bound)
    prof_frac_reco_eta_c = TProfile("frac_reco_eta_c", "Fraction of jets with matched tracks and GNN reconstructible vertices;Jet #eta;Jet fraction",20,-jet_eta_bound,jet_eta_bound)

    #read in normalization constants for features
    mean_features = np.zeros(nnfeatures)
    std_features = np.zeros(nnfeatures)
    if os.path.isfile(normfile_name):
        normfile = open(normfile_name, "r")
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

            hf_pdgids = graph.ndata['track_ancestors'].numpy()[:,1]
            prev_b_pdgids = graph.ndata['track_ancestors'].numpy()[:,3]

            track_flavors = graph.ndata['track_info'].numpy()[:,0]
            jet_flavor = int(graph.ndata['jet_info'].numpy()[0,0]) #jet flavor from truth
            reco_jet_flavor = 0 #jet flavor from assigned track_flavors

            jet_pv = [graph.ndata['jet_info'].numpy()[0,1], graph.ndata['jet_info'].numpy()[0,2], graph.ndata['jet_info'].numpy()[0,3]]
            used_sv = [] #array containing secondary vertices that were already considered for relevant plots

            passed_tracks = 0 
            for i in range(len(features_base[:,0])):
                track_sv = [graph.ndata['track_info'].numpy()[i,1], graph.ndata['track_info'].numpy()[i,2], graph.ndata['track_info'].numpy()[i,3]]
                if passed_cuts[i] == 1:
                    passed_tracks += 1
                    if track_flavors[i] == 1: #fill accepted tracks in b histograms
                        hist_trk_pt_b.Fill(features_base[i,0])
                        hist_trk_theta_b.Fill(features_base[i,1])
                        hist_trk_phi_b.Fill(features_base[i,2])
                        hist_trk_d0_b.Fill(features_base[i,3])
                        hist_trk_z0_b.Fill(features_base[i,4])
                        hist_acc_trk_id_b.Fill(b_pdgids.index(abs(hf_pdgids[i])))
                        hist_acc_trk_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
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
                    elif track_flavors[i] == 3: #fill accepted tracks in btoc histograms
                        hist_trk_pt_btoc.Fill(features_base[i,0])
                        hist_trk_theta_btoc.Fill(features_base[i,1])
                        hist_trk_phi_btoc.Fill(features_base[i,2])
                        hist_trk_d0_btoc.Fill(features_base[i,3])
                        hist_trk_z0_btoc.Fill(features_base[i,4])
                        hist_acc_trk_id_btoc.Fill(b_pdgids.index(abs(prev_b_pdgids[i])))
                        hist_acc_trk_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
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
                    elif track_flavors[i] == 2: #fill accepted tracks in c histogram
                        hist_trk_pt_c.Fill(features_base[i,0])
                        hist_trk_theta_c.Fill(features_base[i,1])
                        hist_trk_phi_c.Fill(features_base[i,2])
                        hist_trk_d0_c.Fill(features_base[i,3])
                        hist_trk_z0_c.Fill(features_base[i,4])
                        hist_acc_trk_id_c.Fill(c_pdgids.index(abs(hf_pdgids[i])))
                        hist_acc_trk_lxy_c.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
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
                    else: #fill accepted tracks in nohf histograms
                        hist_trk_pt_nohf.Fill(features_base[i,0])
                        hist_trk_theta_nohf.Fill(features_base[i,1])
                        hist_trk_phi_nohf.Fill(features_base[i,2])
                        hist_trk_d0_nohf.Fill(features_base[i,3])
                        hist_trk_z0_nohf.Fill(features_base[i,4])
                        if incl_errors:
                            hist_trk_p_err_nohf.Fill(features_errors[i,0])
                            hist_trk_theta_err_nohf.Fill(features_errors[i,1])
                            hist_trk_phi_err_nohf.Fill(features_errors[i,2])
                            hist_trk_d0_err_nohf.Fill(features_errors[i,3])
                            hist_trk_z0_err_nohf.Fill(features_errors[i,4])

                    if track_flavors[i] == 1 or track_flavors[i] == 2 or track_flavors[i] == 3:
                        hist_acc_trk_pt_hf.Fill(abs(1/features_base[i,0]))
                        hist_acc_trk_z0_hf.Fill(features_base[i,4])
                        if incl_vweight: hist_acc_trk_vweight_hf.Fill(features_vweight[i,0])
                    else:
                        hist_acc_trk_pt_nohf.Fill(abs(1/features_base[i,0]))
                        hist_acc_trk_z0_nohf.Fill(features_base[i,4])
                        if incl_vweight: hist_acc_trk_vweight_nohf.Fill(features_vweight[i,0])

                else:
                    if track_flavors[i] == 1:
                        hist_rej_trk_id_b.Fill(b_pdgids.index(abs(hf_pdgids[i])))
                        hist_rej_trk_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                    elif track_flavors[i] == 3:
                        hist_rej_trk_id_btoc.Fill(b_pdgids.index(abs(prev_b_pdgids[i])))
                        hist_rej_trk_lxy_b.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))
                    elif track_flavors[i] == 2:
                        hist_rej_trk_id_c.Fill(c_pdgids.index(abs(hf_pdgids[i])))
                        hist_rej_trk_lxy_c.Fill(np.linalg.norm(np.array(jet_pv)-np.array(track_sv)))

                    if track_flavors[i] == 1 or track_flavors[i] == 2 or track_flavors[i] == 3:
                        hist_rej_trk_pt_hf.Fill(abs(1/features_base[i,0]))
                        hist_rej_trk_z0_hf.Fill(features_base[i,4])
                        if incl_vweight: hist_rej_trk_vweight_hf.Fill(features_vweight[i,0])
                    else:
                        hist_rej_trk_pt_nohf.Fill(abs(1/features_base[i,0]))
                        hist_rej_trk_z0_nohf.Fill(features_base[i,4])
                        if incl_vweight: hist_rej_trk_vweight_nohf.Fill(features_vweight[i,0])

            jet_classification[jet_flavor,reco_jet_flavor] += 1
            if jet_flavor == 1:
                hist_jet_pt_b.Fill(features_base[0,5])
                hist_jet_eta_b.Fill(features_base[0,6])
                hist_jet_phi_b.Fill(features_base[0,7])
                hist_no_trk_b.Fill(passed_tracks)
                prof_frac_reco_pt_b.Fill(features_base[0,5], int(reco_jet_flavor == 1))
                prof_frac_reco_eta_b.Fill(features_base[0,6], int(reco_jet_flavor == 1))
            elif jet_flavor == 2:
                hist_jet_pt_c.Fill(features_base[0,5])
                hist_jet_eta_c.Fill(features_base[0,6])
                hist_jet_phi_c.Fill(features_base[0,7])
                hist_no_trk_c.Fill(passed_tracks)
                prof_frac_reco_pt_c.Fill(features_base[0,5], int(reco_jet_flavor == 2))
                prof_frac_reco_eta_c.Fill(features_base[0,6], int(reco_jet_flavor == 2))
            else:
                hist_jet_pt_l.Fill(features_base[0,5])
                hist_jet_eta_l.Fill(features_base[0,6])
                hist_jet_phi_l.Fill(features_base[0,7])
                hist_no_trk_l.Fill(passed_tracks)

    plot_confusion_matrix(jet_classification, ["l","b","c"], ["l","b","c"], "", ["Target jet classification", "Jet flavor label"], data_path+data_name+"_jet_cm.png")
    plot_hist([hist_trk_pt_b, hist_trk_pt_btoc, hist_trk_pt_c, hist_trk_pt_nohf], ["bH", "bH->cH", "prompt cH", "no HF"], True, False, True, data_path+data_name+"_trk_pt.png", "HIST NOSTACK", scaling=[mean_features[0],std_features[0]])
    plot_hist([hist_trk_theta_b, hist_trk_theta_btoc, hist_trk_theta_c, hist_trk_theta_nohf], ["bH", "bH->cH", "prompt cH", "no HF"], True, False, True, data_path+data_name+"_trk_theta.png", "HIST NOSTACK", scaling=[mean_features[1],std_features[1]])
    plot_hist([hist_trk_phi_b, hist_trk_phi_btoc, hist_trk_phi_c, hist_trk_phi_nohf], ["bH", "bH->cH", "prompt cH", "no HF"], True, False, True, data_path+data_name+"_trk_phi.png", "HIST NOSTACK", scaling=[mean_features[2],std_features[2]])
    plot_hist([hist_trk_d0_b, hist_trk_d0_btoc, hist_trk_d0_c, hist_trk_d0_nohf], ["bH", "bH->cH", "prompt cH", "no HF"], True, False, True, data_path+data_name+"_trk_d0.png", "HIST NOSTACK", scaling=[mean_features[3],std_features[3]])
    plot_hist([hist_trk_z0_b, hist_trk_z0_btoc, hist_trk_z0_c, hist_trk_z0_nohf], ["bH", "bH->cH", "prompt cH", "no HF"], True, False, True, data_path+data_name+"_trk_z0.png", "HIST NOSTACK", scaling=[mean_features[4],std_features[4]])
    plot_hist([hist_jet_pt_b, hist_jet_pt_c, hist_jet_pt_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_jet_pt.png", "HIST NOSTACK", scaling=[mean_features[5],std_features[5]])
    plot_hist([hist_jet_eta_b, hist_jet_eta_c, hist_jet_eta_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_jet_eta.png", "HIST NOSTACK", scaling=[mean_features[6],std_features[6]])
    plot_hist([hist_jet_phi_b, hist_jet_phi_c, hist_jet_phi_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_jet_phi.png", "HIST NOSTACK", scaling=[mean_features[7],std_features[7]])
    plot_hist([hist_no_trk_b, hist_no_trk_c, hist_no_trk_l], ["b", "c", "l"], True, False, True, data_path+data_name+"_no_trk.png", "HIST NOSTACK")
    plot_hist([hist_vertex_lxy_b, hist_vertex_lxy_c], ["b", "c"], True, True, True, data_path+data_name+"_lxy.png", "HIST NOSTACK")
    plot_profile([prof_frac_reco_pt_b, prof_frac_reco_pt_c], ["b", "c"], data_path+data_name+"_frac_reco_pt.png")
    plot_profile([prof_frac_reco_eta_b, prof_frac_reco_eta_c], ["b", "c"], data_path+data_name+"_frac_reco_eta.png")

    if incl_errors:
        plot_hist([hist_trk_p_err_b, hist_trk_p_err_btoc, hist_trk_p_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_p_err.png", "HIST NOSTACK", scaling=[mean_features[8],std_features[8]])
        plot_hist([hist_trk_theta_err_b, hist_trk_theta_err_btoc, hist_trk_theta_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_theta_err.png", "HIST NOSTACK", scaling=[mean_features[9],std_features[9]])
        plot_hist([hist_trk_phi_err_b, hist_trk_phi_err_btoc, hist_trk_phi_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_phi_err.png", "HIST NOSTACK", scaling=[mean_features[10],std_features[10]])
        plot_hist([hist_trk_d0_err_b, hist_trk_d0_err_btoc, hist_trk_d0_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_d0_err.png", "HIST NOSTACK", scaling=[mean_features[11],std_features[11]])
        plot_hist([hist_trk_z0_err_b, hist_trk_z0_err_btoc, hist_trk_z0_err_c], ["bH", "bH->cH", "prompt cH"], True, False, True, data_path+data_name+"_trk_z0_err.png", "HIST NOSTACK", scaling=[mean_features[12],std_features[12]])

    plot_hist([hist_acc_trk_pt_hf, hist_rej_trk_pt_hf], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_pt_hf.png", "HIST NOSTACK")
    plot_hist([hist_acc_trk_pt_nohf, hist_rej_trk_pt_nohf], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_pt_nohf.png", "HIST NOSTACK")
    plot_hist([hist_acc_trk_z0_hf, hist_rej_trk_z0_hf], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_z0_hf.png", "HIST NOSTACK")
    plot_hist([hist_acc_trk_z0_nohf, hist_rej_trk_z0_nohf], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_z0_nohf.png", "HIST NOSTACK")
    plot_hist([hist_acc_trk_lxy_b, hist_rej_trk_lxy_b], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_lxy_b.png", "HIST NOSTACK")
    plot_hist([hist_acc_trk_lxy_c, hist_rej_trk_lxy_c], ["passed", "cut"], False, True, True, data_path+data_name+"_cut_trk_lxy_c.png", "HIST NOSTACK")
    plot_bar([hist_acc_trk_id_b, hist_rej_trk_id_b], b_pdgids, ["passed", "cut"], False, True, data_path+data_name+"_trk_ancid_b.png", "HIST NOSTACK")
    plot_bar([hist_acc_trk_id_c, hist_rej_trk_id_c], c_pdgids, ["passed", "cut"], False, True, data_path+data_name+"_trk_ancid_c.png", "HIST NOSTACK")
    plot_bar([hist_acc_trk_id_btoc, hist_rej_trk_id_btoc], b_pdgids, ["passed", "cut"], False, True, data_path+data_name+"_trk_ancid_btoc.png", "HIST NOSTACK")

    if incl_vweight:
        plot_hist([hist_acc_trk_vweight_hf, hist_rej_trk_vweight_hf], ["passsed", "cut"], False, True, True, data_path+data_name+"_cut_trk_vweight_hf.png", "HIST NOSTACK")
        plot_hist([hist_acc_trk_vweight_nohf, hist_rej_trk_vweight_nohf], ["passsed", "cut"], False, True, True, data_path+data_name+"_cut_trk_vweight_nohf.png", "HIST NOSTACK")


if __name__ == '__main__':
    main(sys.argv)
