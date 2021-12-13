#!/usr/bin/env python

###################################### compare_results.py ######################################
# PURPOSE: generates plots comparing GNN performance to SV1
# EDIT TO: add more plots, update graph structure (if modified in create_graphs)
# ------------------------------------------Summary---------------------------------------------
# This script generates a variety of plots related to the performance of the GNN as compared to
# SV1. This includes plots showing the efficiency and fake rate of both algorithms (see below
# for how this is specifically defined here), the fraction of (in)correctly associated tracks
# for each vertex as well as an ROC curve showing efficiency/fake rate for multiple different
# GNN edge score thresholds (the latter can also serve as an easy comparison between different
# GNN runs). Because of the rather complicated nature of matching GNN predictions to truth SVs
# from MC, this script is likely the least general in the framework and thus generally requires
# the most tweaking.
################################################################################################

#EFFICIENCY / FAKE RATE DEFINITIONS
#Algorithmic efficiency = Number of b/c jets with correctly reconstructed SV / Number of b/c jets with "reconstructable" SV (only calculated for jets with 1 "reconstructable SV")
#Algorithmic fake rate = Number of jets with falsely reconstructed SV / Number of jets with reconstructed SV (only calculated for jets with 1 "reconstructable SV")
#Physics efficiency = Number of b/c jets with reconstructed SV / Number of b/c jets
#Physics fake rate = Number of l jets with reconstructed SV / Number of jets with reconstructed SV


import matplotlib as mpl
mpl.use('Agg')

import dgl
import torch as th
import torch.nn as nn
import scipy.optimize as opt
import scipy.integrate as integrate
import os,sys,math,glob,time
import numpy as np
import ROOT
from ROOT import gROOT, gStyle, TFile, TH1D, TLegend, TCanvas, TProfile, gPad, TGraph, TMultiGraph

from GNN_eval import *
from plot_functions import *
import options


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--runnumber", type=str, default=0, dest="runnumber", help="unique identifier for current run")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="output_dir", help="name of directory where GNN output is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="infile_name", help="name of dataset to train on (without hdf5 extension)")
    parser.add_argument("-n", "--normed", type=int, default=1, dest="use_normed", help="choose whether to use normalized features or not")
    args = parser.parse_args()

    runnumber = args.runnumber
    infile_name = args.infile_name
    infile_path = args.data_dir
    outfile_path = args.output_dir
    norm = args.use_normed

    #import options from option file
    batch_size = options.batch_size
    score_threshold = options.score_threshold
    plot_roc = options.plot_roc
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound
    ntrk_bound = options.ntrk_bound

    graphfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_results.bin"
    paramfile_name = infile_path+infile_name+"_params"
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber
    normfile_name = infile_path+infile_name+"_norm"

    #mapping of flavor labels used - set in create_graphs
    trk_flavor_labels = ['nm','b','c','btoc','p','s','o']

    #calculate number of features in graphs
    sample_graph = (dgl.load_graphs(graphfile_name, [0]))[0][0]
    incl_errors = incl_corr = incl_hits = incl_vweight = False
    nnfeatures_base = sample_graph.ndata['features_base'].size()[1]
    nnfeatures = nnfeatures_base
    if 'features_vweight' in sample_graph.ndata.keys():
        nnfeatures_vweight = sample_graph.ndata['features_vweight'].size()[1]
        incl_vweight = True
        nnfeatures += nnfeatures_vweight
    if 'features_errors' in sample_graph.ndata.keys():
        nnfeatures_errors = sample_graph.ndata['features_errors'].size()[1]
        incl_errors = True
        nnfeatures += nnfeatures_errors
    if 'features_hits' in sample_graph.ndata.keys():
        nnfeatures_hits = sample_graph.ndata['features_hits'].size()[1]
        incl_hits = True
        nnfeatures += nnfeatures_hits
    if 'features_corr' in sample_graph.ndata.keys():
        nnfeatures_corr = sample_graph.ndata['features_corr'].size()[1]
        incl_corr = True
        nnfeatures += nnfeatures_corr

    #read in length of test file
    if os.path.isfile(paramfile_name):
        paramfile = open(paramfile_name, "r")
        train_len = int(float(paramfile.readline()))
        val_len = int(float(paramfile.readline()))
        test_len = int(float(paramfile.readline()))
    else:
        print("ERROR: Specified parameter file not found")
        return 1
    batches = int(math.ceil(test_len/batch_size))

    #read in normalization parameters
    if norm and os.path.isfile(normfile_name):
        normfile = open(normfile_name, "r")
        for i in range(10):
            _ = normfile.readline()
        jet_pt_mean = float(normfile.readline())
        jet_pt_std = float(normfile.readline())
        jet_eta_mean = float(normfile.readline())
        jet_eta_std = float(normfile.readline())
    elif norm:
        print("ERROR: Specified norm file not found")
        return 1

    bin_edges_one = np.linspace(-0.05,1.05,12)
    bin_edges_ntrk = np.linspace(-0.5,ntrk_bound+0.5,ntrk_bound+2)
	
    #histograms of true SV number for each vertex label
    no_true_sv_hist_tot = TH1D("no_true_sv_tot", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges_one[0:7]*10)
    no_true_sv_hist_c = TH1D("no_true_sv_c", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges_one[0:7]*10)
    no_true_sv_hist_b = TH1D("no_true_sv_b", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges_one[0:7]*10)
    no_true_sv_hist_btoc = TH1D("no_true_sv_btoc", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges_one[0:7]*10)

    #histograms of (in)correctly associated tracks to reco vertices for GNN, SV1
    sv1_corrp_hist = TH1D("SV1", "Fraction of correct tracks in reco vertices matched to truth vertices;Fraction of tracks;Fraction of jets",11,bin_edges_one)
    sv1_fakep_hist = TH1D("SV1 ", "Fraction of incorrect tracks in reco vertices matched to truth vertices;Fraction of tracks;Fraction of jets",11,bin_edges_one)
    gnn_corrp_hist = TH1D("GNN", "Fraction of correct tracks in reco vertices matched to truth vertices;Fraction of tracks;Fraction of jets",11,bin_edges_one)
    gnn_fakep_hist = TH1D("GNN ", "Fraction of incorrect tracks in reco vertices matched to truth vertices;Fraction of tracks;Fraction of jets",11,bin_edges_one)
    sv1_pt_profile_corrp_c = TProfile("sv1_pt_corrp_c", "Fraction of correct tracks in truth vertices matched to reco vertices for c jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_corrp_c = TProfile("gnn_pt_corrp_c", "Fraction of correct tracks in truth vertices matched to reco vertices for c jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_corrp_c = TProfile("sv1_eta_corrp_c", "Fraction of correct tracks in truth vertices matched to reco vertices for c jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_corrp_c = TProfile("gnn_eta_corrp_c", "Fraction of correct tracks in truth vertices matched to reco vertices for c jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_corrp_b = TProfile("sv1_pt_corrp_b", "Fraction of correct tracks in truth vertices matched to reco vertices for b jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_corrp_b = TProfile("gnn_pt_corrp_b", "Fraction of correct tracks in truth vertices matched to reco vertices for b jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_corrp_b = TProfile("sv1_eta_corrp_b", "Fraction of correct tracks in truth vertices matched to reco vertices for b jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_corrp_b = TProfile("gnn_eta_corrp_b", "Fraction of correct tracks in truth vertices matched to reco vertices SV for b jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_fakep_c = TProfile("sv1_pt_fakep_c", "Fraction of incorrect tracks in truth vertices matched to reco vertices for c jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fakep_c = TProfile("gnn_pt_fakep_c", "Fraction of incorrect tracks in truth vertices matched to reco vertices for c jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fakep_c = TProfile("sv1_eta_fakep_c", "Fraction of incorrect tracks in truth vertices matched to reco vertices for c jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_fakep_c = TProfile("gnn_eta_fakep_c", "Fraction of incorrect tracks in truth vertices matched to reco vertices for c jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_fakep_b = TProfile("sv1_pt_fakep_b", "Fraction of incorrect tracks in truth vertices matched to reco vertices for b jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fakep_b = TProfile("gnn_pt_fakep_b", "Fraction of incorrect tracks in truth vertices matched to reco vertices for b jets as a function of jet pT;pT [GeV];Fraction of tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fakep_b = TProfile("sv1_eta_fakep_b", "Fraction of incorrect tracks in truth vertices matched to reco vertices for b jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_fakep_b = TProfile("gnn_eta_fakep_b", "Fraction of incorrect tracks in truth vertices matched to reco vertices for b jets as a function of jet eta;eta;Fraction of tracks",20,-jet_eta_bound,jet_eta_bound)
    hist_pc_list = [sv1_corrp_hist, gnn_corrp_hist]
    hist_pf_list = [sv1_fakep_hist, gnn_fakep_hist]

    #algorithmic efficiency/fr histograms - efficiency is only evaluated for jets with exactly one true SV
    sv1_pt_profile_alg_eff_c = TProfile("sv1_pt_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_alg_eff_c = TProfile("gnn_pt_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_alg_eff_c = TProfile("sv1_eta_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_alg_eff_c = TProfile("gnn_eta_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_alg_eff_b = TProfile("sv1_pt_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_alg_eff_b = TProfile("gnn_pt_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_alg_eff_b = TProfile("sv1_eta_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_alg_eff_b = TProfile("gnn_eta_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    sv1_ntrk_profile_alg_eff_c = TProfile("sv1_ntrk_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_alg_eff_c = TProfile("gnn_ntrk_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    sv1_ntrk_profile_alg_eff_b = TProfile("sv1_ntrk_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_alg_eff_b = TProfile("gnn_ntrk_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    sv1_lxy_profile_alg_eff_c = TProfile("sv1_lxy_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs Lxy;Lxy [mm];Efficiency",20,0,100)
    gnn_lxy_profile_alg_eff_c = TProfile("gnn_lxy_alg_eff_c", "SV reconstruction algorithmic efficiency for c jets with a single SV vs Lxy;Lxy [mm];Efficiency",20,0,100)
    sv1_lxy_profile_alg_eff_b = TProfile("sv1_lxy_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs Lxy;Lxy [mm];Efficiency",20,0,100)
    gnn_lxy_profile_alg_eff_b = TProfile("gnn_lxy_alg_eff_b", "SV reconstruction algorithmic efficiency for b jets with a single SV vs Lxy;Lxy [mm];Efficiency",20,0,100)
    sv1_pt_profile_alg_fr = TProfile("sv1_pt_alg_fr", "SV reconstruction algorithmic fake rate vs jet pT;pT [GeV];Fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_alg_fr = TProfile("gnn_pt_alg_fr", "SV reconstruction algorithmic fake rate vs jet pT;pT [GeV];Fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_alg_fr = TProfile("sv1_eta_alg_fr", "SV reconstruction algorithmic fake rate vs jet eta;eta;Fake rate",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_alg_fr = TProfile("gnn_eta_alg_fr", "SV reconstruction algorithmic fake rate vs jet eta;eta;Fake rate",20,-jet_eta_bound,jet_eta_bound)
 
    #physics efficiency/fr histograms
    sv1_pt_profile_phys_eff_c = TProfile("sv1_pt_phys_eff_c", "SV reconstruction physics efficiency for c jets vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_phys_eff_c = TProfile("gnn_pt_phys_eff_c", "SV reconstruction physics efficiency for c jets vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_phys_eff_c = TProfile("sv1_eta_phys_eff_c", "SV reconstruction physics efficiency for c jets vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_phys_eff_c = TProfile("gnn_eta_phys_eff_c", "SV reconstruction physics efficiency for c jets vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_phys_eff_b = TProfile("sv1_pt_phys_eff_b", "SV reconstruction physics efficiency for b jets vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_phys_eff_b = TProfile("gnn_pt_phys_eff_b", "SV reconstruction physics efficiency for b jets vs jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_phys_eff_b = TProfile("sv1_eta_phys_eff_b", "SV reconstruction physics efficiency for b jets vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_phys_eff_b = TProfile("gnn_eta_phys_eff_b", "SV reconstruction physics efficiency for b jets vs jet eta;eta;Efficiency",20,-jet_eta_bound,jet_eta_bound)
    sv1_ntrk_profile_phys_eff_c = TProfile("sv1_ntrk_phys_eff_c", "SV reconstruction physics efficiency for c jets vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_phys_eff_c = TProfile("gnn_ntrk_phys_eff_c", "SV reconstruction physics efficiency for c jets vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    sv1_ntrk_profile_phys_eff_b = TProfile("sv1_ntrk_phys_eff_b", "SV reconstruction physics efficiency for b jets vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_phys_eff_b = TProfile("gnn_ntrk_phys_eff_b", "SV reconstruction physics efficiency for b jets vs track number (post-cuts);Number of tracks;Efficiency",ntrk_bound+1,bin_edges_ntrk)
    sv1_pt_profile_phys_fr = TProfile("sv1_pt_phys_fr", "SV reconstruction physics fake rate as a function of jet pT;pT [GeV];Fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_phys_fr = TProfile("gnn_pt_phys_fr", "SV reconstruction physics fake rate as a function of jet pT;pT [GeV];Fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_phys_fr = TProfile("sv1_eta_phys_fr", "SV reconstruction physics fake rate as a function of jet eta;eta;Fake rate",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_phys_fr = TProfile("gnn_eta_phys_fr", "SV reconstruction physics fake rate as a function of jet eta;eta;Fake rate",20,-jet_eta_bound,jet_eta_bound)
    
    #histograms showing track flavor label prevalence in falsely associated tracks
    gnn_trk_assoc_hist_fake = TH1D("gnn_trk_assoc_fake", "Origin of tracks in fake GNN vertices;Track flavor label;Number of tracks", len(trk_flavor_labels), 0, len(trk_flavor_labels))
    gnn_trk_assoc_hist_true = TH1D("gnn_trk_assoc_true", "Origin of misassociated tracks in true GNN vertices;Track flavor label; Number of tracks", len(trk_flavor_labels), 0, len(trk_flavor_labels))

    #initialize matrices for overall SV predictions and plots
    gnn_cm = np.zeros((3,3), dtype=int)
    sv1_cm = np.zeros((3,3), dtype=int)
    b_tot_alg = b_found_gnn_alg = b_found_sv1_alg = c_tot_alg = c_found_gnn_alg = c_found_sv1_alg = fake_found_gnn_alg = fake_found_sv1_alg = tot_pred_gnn_alg = tot_pred_sv1_alg = 0
    b_tot_phys = b_found_gnn_phys = b_found_sv1_phys = c_tot_phys = c_found_gnn_phys = c_found_sv1_phys = fake_found_gnn_phys = fake_found_sv1_phys = tot_pred_gnn_phys = tot_pred_sv1_phys = 0

    #evaluate run statistics
    for ibatch in range(batches):
        
        #calculate batch indices
        istart = ibatch*batch_size
        if ibatch == (batches-1) and test_len%batch_size != 0:
            iend = istart + (test_len%batch_size)
        else:
            iend = (ibatch+1)*batch_size

        #load batch from file
        batch = dgl.batch(dgl.load_graphs(graphfile_name, list(range(istart, iend)))[0])
        g_list = dgl.unbatch(batch)

        for g in g_list:
            pv_coord = np.array([g.ndata['jet_info'][0,1], g.ndata['jet_info'][0,2], g.ndata['jet_info'][0,3]])
            jet_pt = g.ndata['features_base'][0,5]
            jet_eta = g.ndata['features_base'][0,6]
            if norm:
                jet_pt = jet_pt*jet_pt_std+jet_pt_mean
                jet_eta = jet_eta*jet_eta_std+jet_eta_mean

            gnn_vertices = find_vertices_bin(g, 'gnn', score_threshold)
            true_vertices = find_vertices_bin(g, 'truth', 1.0)
            sv1_vertex = np.argwhere(g.ndata['reco_use'].cpu().numpy().astype(int)[:,1]).flatten()
            jet_flavor = g.ndata['jet_info'].cpu().numpy().astype(int)[0,0]
            ntrk = g.num_nodes()
            if sv1_vertex.size > 0: sv1_vertex = [sv1_vertex]

            jet_gnn_cm, gnn_vertex_metrics = compare_vertices(true_vertices, gnn_vertices)
            jet_sv1_cm, sv1_vertex_metrics = compare_vertices(true_vertices, sv1_vertex)
            gnn_cm += jet_gnn_cm
            sv1_cm += jet_sv1_cm

            no_true_sv_hist_tot.Fill(len(true_vertices))

            #for each GNN reco vertex, fill category histograms for falsely associated tracks and metric histograms
            gnn_true_vertex_assoc = associate_vertices(gnn_vertex_metrics, 'r')
            for i, vertex in enumerate(gnn_vertices):
                if gnn_true_vertex_assoc[i] > -1:
                    hist_pc_list[1].Fill(gnn_vertex_metrics[gnn_true_vertex_assoc[i],i,1])
                    hist_pf_list[1].Fill(gnn_vertex_metrics[gnn_true_vertex_assoc[i],i,2])
                else:
                    hist_pc_list[1].Fill(0)
                    hist_pf_list[1].Fill(1)

                for track in vertex:
                    trk_label = g.ndata['track_info'][track,0].numpy()
                    if gnn_true_vertex_assoc[i] == -1: #vertex association not available - fake GNN vertex
                        gnn_trk_assoc_hist_fake.Fill(trk_label)
                    elif track not in true_vertices[gnn_true_vertex_assoc[i]]:
                        gnn_trk_assoc_hist_true.Fill(trk_label)

            #for each SV1 reco vertex, fill category histograms for falsely associated tracks and metric histograms
            sv1_true_vertex_assoc = associate_vertices(sv1_vertex_metrics, 'r')
            for i, vertex in enumerate(sv1_vertex):
                if sv1_true_vertex_assoc[i] > -1:
                    hist_pc_list[0].Fill(sv1_vertex_metrics[sv1_true_vertex_assoc[i],i,1])
                    hist_pf_list[0].Fill(sv1_vertex_metrics[sv1_true_vertex_assoc[i],i,2])
                else:
                    hist_pc_list[0].Fill(0)
                    hist_pf_list[0].Fill(1)

            no_b = no_c = 0
            #for each true vertex, fill efficiency/fake rate histograms for best matching GNN/SV1 vertex
            true_gnn_vertex_assoc = associate_vertices(gnn_vertex_metrics, 't')
            true_sv1_vertex_assoc = associate_vertices(sv1_vertex_metrics, 't')
            for i, true_vertex in enumerate(true_vertices):
                edge_id = g.edge_ids(true_vertex[0],true_vertex[1])
                sv_coord = np.array([g.ndata['track_info'][true_vertex[0],1], g.ndata['track_info'][true_vertex[0],2], g.ndata['track_info'][true_vertex[0],3]])
                Lxy = np.linalg.norm(pv_coord-sv_coord)
                vertex_flavor = g.ndata['track_info'][true_vertex[0],0] #determine vertex flavor from track label
                if vertex_flavor == 1 or vertex_flavor == 3:
                    no_b += 1
                elif vertex_flavor == 2:
                    no_c += 1

                if true_gnn_vertex_assoc[i] > -1:
                    if vertex_flavor == 2:
                        gnn_pt_profile_corrp_c.Fill(jet_pt, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],1])
                        gnn_pt_profile_fakep_c.Fill(jet_pt, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],2])
                        gnn_eta_profile_corrp_c.Fill(jet_eta, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],1])
                        gnn_eta_profile_fakep_c.Fill(jet_eta, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],2])
                    elif vertex_flavor == 1 or vertex_flavor == 3:
                        gnn_pt_profile_corrp_b.Fill(jet_pt, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],1])
                        gnn_pt_profile_fakep_b.Fill(jet_pt, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],2])
                        gnn_eta_profile_corrp_b.Fill(jet_eta, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],1])
                        gnn_eta_profile_fakep_b.Fill(jet_eta, gnn_vertex_metrics[i,true_gnn_vertex_assoc[i],2])

                if true_sv1_vertex_assoc[i] > -1:
                    if vertex_flavor == 2:
                        sv1_pt_profile_corrp_c.Fill(jet_pt, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],1])
                        sv1_pt_profile_fakep_c.Fill(jet_pt, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],2])
                        sv1_eta_profile_corrp_c.Fill(jet_eta, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],1])
                        sv1_eta_profile_fakep_c.Fill(jet_eta, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],2])
                    elif vertex_flavor == 1 or vertex_flavor == 3:
                        sv1_pt_profile_corrp_b.Fill(jet_pt, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],1])
                        sv1_pt_profile_fakep_b.Fill(jet_pt, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],2])
                        sv1_eta_profile_corrp_b.Fill(jet_eta, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],1])
                        sv1_eta_profile_fakep_b.Fill(jet_eta, sv1_vertex_metrics[i,true_sv1_vertex_assoc[i],2])

            if no_b > 0: no_true_sv_hist_b.Fill(no_b)
            if no_c > 0: no_true_sv_hist_c.Fill(no_c)

            #fill algorithmic efficiency histograms for b/c-jets (only jets with exactly 1 true SV)
            if no_b == 1 and no_c == 0:
                b_tot_alg += 1
                b_found_sv1_alg += jet_sv1_cm[1,1]+jet_sv1_cm[1,2]
                b_found_gnn_alg += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]
                sv1_pt_profile_alg_eff_b.Fill(jet_pt, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_pt_profile_alg_eff_b.Fill(jet_pt, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
                sv1_eta_profile_alg_eff_b.Fill(jet_eta, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_eta_profile_alg_eff_b.Fill(jet_eta, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
                sv1_ntrk_profile_alg_eff_b.Fill(ntrk, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_ntrk_profile_alg_eff_b.Fill(ntrk, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
                sv1_lxy_profile_alg_eff_b.Fill(Lxy,float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_lxy_profile_alg_eff_b.Fill(Lxy,float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
            elif no_c == 1 and no_b == 0:
                c_tot_alg += 1
                c_found_sv1_alg += jet_sv1_cm[1,1]+jet_sv1_cm[1,2]
                c_found_gnn_alg += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]
                sv1_pt_profile_alg_eff_c.Fill(jet_pt, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_pt_profile_alg_eff_c.Fill(jet_pt, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
                sv1_eta_profile_alg_eff_c.Fill(jet_eta, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_eta_profile_alg_eff_c.Fill(jet_eta, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
                sv1_ntrk_profile_alg_eff_c.Fill(ntrk, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_ntrk_profile_alg_eff_c.Fill(ntrk, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))
                sv1_lxy_profile_alg_eff_c.Fill(Lxy,float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]))
                gnn_lxy_profile_alg_eff_c.Fill(Lxy,float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]))

            #fill algorithmic fake rate histograms
            if (jet_sv1_cm[0,1]+jet_sv1_cm[0,2]+jet_sv1_cm[1,1]+jet_sv1_cm[1,2]) == 1:
                tot_pred_sv1_alg += 1
                fake_found_sv1_alg += float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2])
                sv1_pt_profile_alg_fr.Fill(jet_pt, float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2]))
                sv1_eta_profile_alg_fr.Fill(jet_eta, float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2]))
            if (jet_gnn_cm[0,1]+jet_gnn_cm[0,2]+jet_gnn_cm[1,1]+jet_gnn_cm[1,2]) == 1:
                tot_pred_gnn_alg += 1
                fake_found_gnn_alg += float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2])
                gnn_pt_profile_alg_fr.Fill(jet_pt, float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2]))
                gnn_eta_profile_alg_fr.Fill(jet_eta, float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2])) 

            #fill physics efficiency histograms for b/c-jets
            if jet_flavor == 1:
                b_tot_phys += 1
                b_found_sv1_phys += float(len(sv1_vertex) != 0)
                b_found_gnn_phys += float(len(gnn_vertices) != 0)
                sv1_pt_profile_phys_eff_b.Fill(jet_pt, float(len(sv1_vertex) != 0))
                gnn_pt_profile_phys_eff_b.Fill(jet_pt, float(len(gnn_vertices) != 0))
                sv1_eta_profile_phys_eff_b.Fill(jet_eta, float(len(sv1_vertex) != 0))
                gnn_eta_profile_phys_eff_b.Fill(jet_eta, float(len(gnn_vertices) != 0))
                sv1_ntrk_profile_phys_eff_b.Fill(ntrk, float(len(sv1_vertex) != 0))
                gnn_ntrk_profile_phys_eff_b.Fill(ntrk, float(len(gnn_vertices) != 0))
            elif jet_flavor == 2:
                c_tot_phys += 1
                c_found_sv1_phys += float(len(sv1_vertex) != 0)
                c_found_gnn_phys += float(len(gnn_vertices) != 0)
                sv1_pt_profile_phys_eff_c.Fill(jet_pt, float(len(sv1_vertex) != 0))
                gnn_pt_profile_phys_eff_c.Fill(jet_pt, float(len(gnn_vertices) != 0))
                sv1_eta_profile_phys_eff_c.Fill(jet_eta, float(len(sv1_vertex) != 0))
                gnn_eta_profile_phys_eff_c.Fill(jet_eta, float(len(gnn_vertices) != 0))
                sv1_ntrk_profile_phys_eff_c.Fill(ntrk, float(len(sv1_vertex) != 0))
                gnn_ntrk_profile_phys_eff_c.Fill(ntrk, float(len(gnn_vertices) != 0))
                        
            #fill physics fake rate histograms
            if (float(len(sv1_vertex) > 0)):
                tot_pred_sv1_phys += 1
                fake_found_sv1_phys += float(jet_flavor == 0)
                sv1_pt_profile_phys_fr.Fill(jet_pt, float(jet_flavor == 0))
                sv1_eta_profile_phys_fr.Fill(jet_eta, float(jet_flavor == 0))
            if (float(len(gnn_vertices) > 0)):
                tot_pred_gnn_phys += 1
                fake_found_gnn_phys += float(jet_flavor == 0)
                gnn_pt_profile_phys_fr.Fill(jet_pt, float(jet_flavor == 0))
                gnn_eta_profile_phys_fr.Fill(jet_eta, float(jet_flavor == 0))

    print('Secondary Vertex prediction results per jet (GNN/SV1):')
    print('             ||  Pred no SV   |   Pred 1 SV   |   Pred >1 SV  |')
    print('---------------------------------------------------------------')
    print(f'Actual no SV || {gnn_cm[0,0]:5} /{sv1_cm[0,0]:6} | {gnn_cm[0,1]:5} /{sv1_cm[0,1]:6} | {gnn_cm[0,2]:5} /{sv1_cm[0,2]:6} |')
    print(f'Actual 1 SV  || {gnn_cm[1,0]:5} /{sv1_cm[1,0]:6} | {gnn_cm[1,1]:5} /{sv1_cm[1,1]:6} | {gnn_cm[1,2]:5} /{sv1_cm[1,2]:6} |')
    print(f'Actual >1 SV || {gnn_cm[2,0]:5} /{sv1_cm[2,0]:6} | {gnn_cm[2,1]:5} /{sv1_cm[2,1]:6} | {gnn_cm[2,2]:5} /{sv1_cm[2,2]:6} |')
    print('---------------------------------------------------------------')
    print(f'Edge score threshold: {score_threshold}')
    with np.errstate(divide='ignore'): #ignore divide by zero error for printing
        print(f'Global algorithmic b-jet efficiency: {b_found_gnn_alg/b_tot_alg:4} (GNN), {b_found_sv1_alg/b_tot_alg:4} (SV1)')
        print(f'Global algorithmic c-jet efficiency: {c_found_gnn_alg/c_tot_alg:4} (GNN), {c_found_sv1_alg/c_tot_alg:4} (SV1)')
        print(f'Global algorithmic fake rate: {np.divide(fake_found_gnn_alg,tot_pred_gnn_alg):4} (GNN), {np.divide(fake_found_sv1_alg,tot_pred_sv1_alg):4} (SV1)')
        print(f'Global physics b-jet efficiency: {b_found_gnn_phys/b_tot_phys:4} (GNN), {b_found_sv1_phys/b_tot_phys:4} (SV1)')
        print(f'Global physics c-jet efficiency: {c_found_gnn_phys/c_tot_phys:4} (GNN), {c_found_sv1_phys/c_tot_phys:4} (SV1)')
        print(f'Global physics fake rate: {np.divide(fake_found_gnn_phys,tot_pred_gnn_phys):4} (GNN), {np.divide(fake_found_sv1_phys,tot_pred_sv1_phys):4} (SV1)')

    plot_profile([sv1_pt_profile_alg_eff_c, gnn_pt_profile_alg_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_pt_c.png")
    plot_profile([sv1_pt_profile_alg_eff_b, gnn_pt_profile_alg_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_pt_b.png")
    plot_profile([sv1_eta_profile_alg_eff_c, gnn_eta_profile_alg_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_eta_c.png")
    plot_profile([sv1_eta_profile_alg_eff_b, gnn_eta_profile_alg_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_eta_b.png")
    plot_profile([sv1_lxy_profile_alg_eff_b, gnn_lxy_profile_alg_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_lxy_b.png")
    plot_profile([sv1_lxy_profile_alg_eff_c, gnn_lxy_profile_alg_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_lxy_c.png")
    plot_profile([sv1_ntrk_profile_alg_eff_b, gnn_ntrk_profile_alg_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_ntrk_b.png")
    plot_profile([sv1_ntrk_profile_alg_eff_c, gnn_ntrk_profile_alg_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_eff_ntrk_c.png")
    plot_profile([sv1_pt_profile_phys_eff_c, gnn_pt_profile_phys_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_eff_pt_c.png")
    plot_profile([sv1_pt_profile_phys_eff_b, gnn_pt_profile_phys_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_eff_pt_b.png")
    plot_profile([sv1_eta_profile_phys_eff_c, gnn_eta_profile_phys_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_eff_eta_c.png")
    plot_profile([sv1_eta_profile_phys_eff_b, gnn_eta_profile_phys_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_eff_eta_b.png")
    plot_profile([sv1_ntrk_profile_phys_eff_b, gnn_ntrk_profile_phys_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_eff_ntrk_b.png")
    plot_profile([sv1_ntrk_profile_phys_eff_c, gnn_ntrk_profile_phys_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_eff_ntrk_c.png")
    plot_profile([sv1_pt_profile_alg_fr, gnn_pt_profile_alg_fr], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_fr_pt.png")
    plot_profile([sv1_eta_profile_alg_fr, gnn_eta_profile_alg_fr], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_alg_fr_eta.png")
    plot_profile([sv1_pt_profile_phys_fr, gnn_pt_profile_phys_fr], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_fr_pt.png")
    plot_profile([sv1_eta_profile_phys_fr, gnn_eta_profile_phys_fr], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_phys_fr_eta.png")
    plot_profile([sv1_pt_profile_corrp_c, gnn_pt_profile_corrp_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_pt_c.png")
    plot_profile([sv1_pt_profile_corrp_b, gnn_pt_profile_corrp_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_pt_b.png")
    plot_profile([sv1_eta_profile_corrp_c, gnn_eta_profile_corrp_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_eta_c.png")
    plot_profile([sv1_eta_profile_corrp_b, gnn_eta_profile_corrp_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_eta_b.png")
    plot_profile([sv1_pt_profile_fakep_c, gnn_pt_profile_fakep_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_pt_c.png")
    plot_profile([sv1_pt_profile_fakep_b, gnn_pt_profile_fakep_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_pt_b.png")
    plot_profile([sv1_eta_profile_fakep_c, gnn_eta_profile_fakep_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_eta_c.png")
    plot_profile([sv1_eta_profile_fakep_b, gnn_eta_profile_fakep_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_eta_b.png")
    
    plot_bar([gnn_trk_assoc_hist_fake], trk_flavor_labels, [], False, True, outfile_name+"_trk_assoc_fake.png", "HIST")
    plot_bar([gnn_trk_assoc_hist_true], trk_flavor_labels, [], False, True, outfile_name+"_trk_assoc_true.png", "HIST")

    ext = ["_corrp.png", "_fakep.png"]
    hist_list_list = [hist_pc_list, hist_pf_list]
    for i in range(len(hist_list_list)):
        plot_metric_hist(hist_list_list[i], [0.0,1.4], outfile_name+ext[i])

    #evaluate roc data
    if plot_roc:
        score_threshold_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        c_alg_efficiency = np.zeros(len(score_threshold_array))
        b_alg_efficiency = np.zeros(len(score_threshold_array))
        alg_fake_rate_num = np.zeros(len(score_threshold_array))
        alg_fake_rate_denom = np.zeros(len(score_threshold_array))
        c_phys_efficiency = np.zeros(len(score_threshold_array))
        b_phys_efficiency = np.zeros(len(score_threshold_array))
        phys_fake_rate_num = np.zeros(len(score_threshold_array))
        phys_fake_rate_denom = np.zeros(len(score_threshold_array))

        for ibatch in range(batches):
            #calculate batch indices
            istart = ibatch*batch_size
            if ibatch == (batches-1) and test_len%batch_size != 0:
                iend = istart + (test_len%batch_size)
            else:
                iend = (ibatch+1)*batch_size

            #load batch from file
            batch = dgl.batch(dgl.load_graphs(graphfile_name, list(range(istart, iend)))[0])
            g_list = dgl.unbatch(batch)

            for ist in range(len(score_threshold_array)):
                score_threshold = score_threshold_array[ist]

                for g in g_list:
                    jet_flavor = g.ndata['jet_info'].cpu().numpy().astype(int)[0,0]
                    gnn_vertices = find_vertices_bin(g, 'gnn', score_threshold)
                    true_vertices = find_vertices_bin(g, 'truth', 1.0)
                    jet_gnn_cm, gnn_vertex_metrics = compare_vertices(true_vertices, gnn_vertices)

                    no_b = no_c = 0
                    for i, true_vertex in enumerate(true_vertices):
                        edge_id = g.edge_ids(true_vertex[0],true_vertex[1])
                        vertex_flavor = g.ndata['track_info'][true_vertex[0],0]
                        if vertex_flavor == 1 or vertex_flavor == 3:
                            no_b += 1
                        elif vertex_flavor == 2:
                            no_c += 1

                    #fill algorithmic efficiency for b/c-jets (only jets with exactly 1 true SV)
                    if no_b == 1 and no_c == 0:
                        b_alg_efficiency[ist] += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]
                    elif no_c == 1 and no_b == 0:
                        c_alg_efficiency[ist] += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]

                    #fill algorithmic fake rate histograms (only jets with exactly 1 true SV)
                    if (jet_gnn_cm[0,1]+jet_gnn_cm[0,2]+jet_gnn_cm[1,1]+jet_gnn_cm[1,2]) == 1:
                        alg_fake_rate_num[ist] += jet_gnn_cm[0,1]+jet_gnn_cm[0,2]
                        alg_fake_rate_denom[ist] += 1

                    #fill physics efficiency histograms for b/c-jets
                    if jet_flavor == 1:
                        b_phys_efficiency[ist] += float(len(gnn_vertices) != 0)
                    elif jet_flavor == 2:
                        c_phys_efficiency[ist] += float(len(gnn_vertices) != 0)

                    if (float(len(gnn_vertices) > 0)):
                        phys_fake_rate_denom[ist] += 1
                        phys_fake_rate_num[ist] += float(jet_flavor == 0)

        b_alg_efficiency = b_alg_efficiency/b_tot_alg
        c_alg_efficiency = c_alg_efficiency/c_tot_alg
        alg_fake_rate = np.divide(alg_fake_rate_num,alg_fake_rate_denom)
        sv1_b_alg_efficiency = np.array([b_found_sv1_alg/b_tot_alg])
        sv1_c_alg_efficiency = np.array([c_found_sv1_alg/c_tot_alg])
        sv1_alg_fake_rate = np.array([fake_found_sv1_alg/tot_pred_sv1_alg])

        b_phys_efficiency = b_phys_efficiency/b_tot_phys
        c_phys_efficiency = c_phys_efficiency/c_tot_phys
        phys_fake_rate = np.divide(phys_fake_rate_num,phys_fake_rate_denom)
        sv1_b_phys_efficiency = np.array([b_found_sv1_phys/b_tot_phys])
        sv1_c_phys_efficiency = np.array([c_found_sv1_phys/c_tot_phys])
        sv1_phys_fake_rate = np.array([fake_found_sv1_phys/tot_pred_sv1_phys])

        alg_roc_curve_b = TGraph(len(b_alg_efficiency), alg_fake_rate, b_alg_efficiency)
        alg_roc_curve_c = TGraph(len(c_alg_efficiency), alg_fake_rate, c_alg_efficiency)
        sv1_alg_eff_b = TGraph(len(sv1_b_alg_efficiency), sv1_alg_fake_rate, sv1_b_alg_efficiency)
        sv1_alg_eff_c = TGraph(len(sv1_c_alg_efficiency), sv1_alg_fake_rate, sv1_c_alg_efficiency)
        plot_roc_curve(alg_roc_curve_b, alg_roc_curve_c, sv1_alg_eff_b, sv1_alg_eff_c, [0.0,1.0], [0.,1.0], "algorithmic", outfile_name+"_alg_roc.png")

        phys_roc_curve_b = TGraph(len(b_phys_efficiency), phys_fake_rate, b_phys_efficiency)
        phys_roc_curve_c = TGraph(len(c_phys_efficiency), phys_fake_rate, c_phys_efficiency)
        sv1_phys_eff_b = TGraph(len(sv1_b_phys_efficiency), sv1_phys_fake_rate, sv1_b_phys_efficiency)
        sv1_phys_eff_c = TGraph(len(sv1_c_phys_efficiency), sv1_phys_fake_rate, sv1_c_phys_efficiency)
        plot_roc_curve(phys_roc_curve_b, phys_roc_curve_c, sv1_phys_eff_b, sv1_phys_eff_c, [0.0,1.0], [0.,1.0], "physics", outfile_name+"_phys_roc.png")

        #calculate area under ROC curve
        def f_fit(x,a,b,c,d,e): return a*x**4+b*x**3+c*x**2+d*x+e #4th order polynomial fitting function
        alg_roc_curve_b_fit = opt.curve_fit(f_fit, alg_fake_rate, b_alg_efficiency)
        alg_roc_curve_b_auc = integrate.quad(lambda x: f_fit(x, alg_roc_curve_b_fit[0], alg_roc_curve_b_fit[1], alg_roc_curve_b_fit[2], alg_roc_curve_b_fit[3], alg_roc_curve_b_fit[4], alg_roc_curve_b_fit[5]), 0, 1)
        alg_roc_curve_c_fit = opt.curve_fit(f_fit, alg_fake_rate, c_alg_efficiency)
        alg_roc_curve_c_auc = integrate.quad(lambda x: f_fit(x, alg_roc_curve_c_fit[0], alg_roc_curve_c_fit[1], alg_roc_curve_c_fit[2], alg_roc_curve_c_fit[3], alg_roc_curve_c_fit[4], alg_roc_curve_c_fit[5]), 0, 1)
        phys_roc_curve_b_fit = opt.curve_fit(f_fit, phys_fake_rate, b_phys_efficiency)
        phys_roc_curve_b_auc = integrate.quad(lambda x: f_fit(x, phys_roc_curve_b_fit[0], phys_roc_curve_b_fit[1], phys_roc_curve_b_fit[2], phys_roc_curve_b_fit[3], phys_roc_curve_b_fit[4], phys_roc_curve_b_fit[5]), 0, 1)
        phys_roc_curve_c_fit = opt.curve_fit(f_fit, phys_fake_rate, c_phys_efficiency)
        phys_roc_curve_c_auc = integrate.quad(lambda x: f_fit(x, phys_roc_curve_c_fit[0], phys_roc_curve_c_fit[1], phys_roc_curve_c_fit[2], phys_roc_curve_c_fit[3], phys_roc_curve_c_fit[4], phys_roc_curve_c_fit[5]), 0, 1)
        print("AuC for algorithmic b: {}".format(alg_roc_curve_b_auc))
        print("AuC for algorithmic c: {}".format(alg_roc_curve_c_auc))
        print("AuC for physics b: {}".format(phys_roc_curve_b_auc))
        print("AuC for physics c: {}".format(phys_roc_curve_c_auc))

    canv1 = TCanvas("c1", "c1", 800, 600)

    gStyle.SetOptStat(0)
    gPad.SetLogy()
    hist_list = [no_true_sv_hist_tot, no_true_sv_hist_b, no_true_sv_hist_c]
    colorlist = [1,4,2,8]
    labellist = ['total', 'bH', 'prompt cH']
    legend = TLegend(0.76,0.88-0.08*len(hist_list),0.91,0.88,'','NDC')
    for i in range(len(hist_list)):
        hist_list[i].SetLineColorAlpha(colorlist[i],0.65)
        legend.AddEntry(hist_list[i], "#splitline{%s}{#splitline{%d entries}{mean=%.2f}}"%(labellist[i], hist_list[i].GetEntries(), hist_list[i].GetMean()), "l")
        if i == 0: hist_list[i].Draw()
        else: hist_list[i].Draw("SAMES E1")
    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    canv1.SaveAs(outfile_name+"_no_sv.png")
    canv1.Clear()
    del canv1


if __name__ == '__main__':
    main(sys.argv)
