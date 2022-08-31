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
import os,sys,math,glob,time
import numpy as np
import ROOT
from ROOT import gROOT, gStyle, TFile, TH1D, TLegend, TCanvas, TProfile, gPad, TGraph, TMultiGraph

from GNN_eval import *
from plot_functions import *


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--runnumber", type=str, default=0, dest="runnumber", help="unique identifier for current run")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="output_dir", help="name of directory where GNN output is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="infile_name", help="name of dataset to train on (without hdf5 extension)")
    parser.add_argument("-f", "--options", type=str, required=True, dest="option_file", help="name of file containing script options")
    args = parser.parse_args()

    runnumber = args.runnumber
    infile_name = args.infile_name
    infile_path = args.data_dir
    outfile_path = args.output_dir
    option_file = args.option_file

    options = __import__(option_file, globals(), locals(), [], 0)

    #import options from option file
    batch_size = options.batch_size
    score_threshold = options.score_threshold
    plot_roc = options.plot_roc
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound
    ntrk_bound = options.ntrk_bound
    cut_string = options.cut_string

    graphfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_results.bin"
    paramfile_name = infile_path+infile_name+"_params"
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber

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

    bin_edges_one = np.linspace(-0.05,1.05,12)
    bin_edges_ntrk = np.linspace(-0.5,ntrk_bound+0.5,ntrk_bound+2)
	
    #histograms of true SV number for each vertex label
    no_true_sv_hist_tot = TH1D("no_true_sv_tot", ";Number of true SV's per jet;Number of jets",6,bin_edges_one[0:7]*10)
    no_true_sv_hist_c = TH1D("no_true_sv_c", ";Number of true SV's per jet;Number of jets",6,bin_edges_one[0:7]*10)
    no_true_sv_hist_b = TH1D("no_true_sv_b", ";Number of true SV's per jet;Number of jets",6,bin_edges_one[0:7]*10)
    no_true_sv_hist_btoc = TH1D("no_true_sv_btoc", ";Number of true SV's per jet;Number of jets",6,bin_edges_one[0:7]*10)

    #histograms of (in)correctly associated tracks to reco vertices for GNN, SV1
    sv1_corrp_hist = TH1D("SV1", ";Fraction of tracks in reco SV correctly matched to truth SV;Fraction of jets",11,bin_edges_one)
    sv1_fakep_hist = TH1D("SV1 ", ";Fraction of tracks in reco SV incorrectly matched to truth SV;Fraction of jets",11,bin_edges_one)
    gnn_corrp_hist = TH1D("GNN", ";Fraction of tracks in reco SV correctly matched to truth SV;Fraction of jets",11,bin_edges_one)
    gnn_fakep_hist = TH1D("GNN ", ";Fraction of tracks in reco SV correctly matched to truth SV;Fraction of jets",11,bin_edges_one)
    sv1_pt_profile_corrp_c = TProfile("sv1_pt_corrp_c", ";pT [GeV];Fraction of tracks from truth SV in reco SV (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_corrp_c = TProfile("gnn_pt_corrp_c", ";pT [GeV];Fraction of tracks from truth SV in reco SV (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_corrp_c = TProfile("sv1_eta_corrp_c", ";#eta;Fraction of tracks from truth SV in reco SV (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_corrp_c = TProfile("gnn_eta_corrp_c", ";#eta;Fraction of tracks from truth SV in reco SV (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_corrp_b = TProfile("sv1_pt_corrp_b", ";pT [GeV];Fraction of tracks from truth SV in reco SV (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_corrp_b = TProfile("gnn_pt_corrp_b", ";pT [GeV];Fraction of tracks from truth SV in reco SV (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_corrp_b = TProfile("sv1_eta_corrp_b", ";#eta;Fraction of tracks from truth SV in reco SV (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_corrp_b = TProfile("gnn_eta_corrp_b", ";#eta;Fraction of tracks from truth SV in reco SV (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_fakep_c = TProfile("sv1_pt_fakep_c", ";pT [GeV];Fraction of tracks from truth SV not in reco SV (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fakep_c = TProfile("gnn_pt_fakep_c", ";pT [GeV];Fraction of tracks from truth SV not in reco SV (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fakep_c = TProfile("sv1_eta_fakep_c", ";#eta;Fraction of tracks from truth SV not in reco SV (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_fakep_c = TProfile("gnn_eta_fakep_c", ";#eta;Fraction of tracks from truth SV not in reco SV (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_fakep_b = TProfile("sv1_pt_fakep_b", ";pT [GeV];Fraction of tracks from truth SV not in reco SV (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fakep_b = TProfile("gnn_pt_fakep_b", ";pT [GeV];Fraction of tracks from truth SV not in reco SV (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fakep_b = TProfile("sv1_eta_fakep_b", ";#eta;Fraction of tracks from truth SV not in reco SV (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_fakep_b = TProfile("gnn_eta_fakep_b", ";#eta;Fraction of tracks from truth SV not in reco SV (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    hist_pc_list = [gnn_corrp_hist, sv1_corrp_hist]
    hist_pf_list = [gnn_fakep_hist, sv1_fakep_hist]

    #algorithmic efficiency/fr histograms - efficiency is only evaluated for jets with exactly one true SV
    sv1_pt_profile_alg_eff_c = TProfile("sv1_pt_alg_eff_c", ";pT [GeV];Algorithmic efficiency (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_alg_eff_c = TProfile("gnn_pt_alg_eff_c", ";pT [GeV];Algorithmic efficiency (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_alg_eff_c = TProfile("sv1_eta_alg_eff_c", ";#eta;Algorithmic efficiency (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_alg_eff_c = TProfile("gnn_eta_alg_eff_c", ";#eta;Algorithmic efficiency (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_alg_eff_b = TProfile("sv1_pt_alg_eff_b", ";pT [GeV];Algorithmic efficiency (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_alg_eff_b = TProfile("gnn_pt_alg_eff_b", ";pT [GeV];Algorithmic efficiency (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_alg_eff_b = TProfile("sv1_eta_alg_eff_b", ";#eta;Algorithmic efficiency (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_alg_eff_b = TProfile("gnn_eta_alg_eff_b", ";#eta;Algorithmic efficiency (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_ntrk_profile_alg_eff_c = TProfile("sv1_ntrk_alg_eff_c", ";Number of tracks;Algorithmic efficiency (c-jets)",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_alg_eff_c = TProfile("gnn_ntrk_alg_eff_c", ";Number of tracks;Algorithmic efficiency (c-jets)",ntrk_bound+1,bin_edges_ntrk)
    sv1_ntrk_profile_alg_eff_b = TProfile("sv1_ntrk_alg_eff_b", ";Number of tracks;Algorithmic efficiency (b-jets)",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_alg_eff_b = TProfile("gnn_ntrk_alg_eff_b", ";Number of tracks;Algorithmic efficiency (b-jets)",ntrk_bound+1,bin_edges_ntrk)
    sv1_lxy_profile_alg_eff_c = TProfile("sv1_lxy_alg_eff_c", ";Lxy [mm];Algorithmic efficiency (c-jets)",20,0,100)
    gnn_lxy_profile_alg_eff_c = TProfile("gnn_lxy_alg_eff_c", ";Lxy [mm];Algorithmic efficiency (c-jets)",20,0,100)
    sv1_lxy_profile_alg_eff_b = TProfile("sv1_lxy_alg_eff_b", ";Lxy [mm];Algorithmic efficiency (b-jets)",20,0,100)
    gnn_lxy_profile_alg_eff_b = TProfile("gnn_lxy_alg_eff_b", ";Lxy [mm];Algorithmic efficiency (b-jets)",20,0,100)
    sv1_pt_profile_alg_fr = TProfile("sv1_pt_alg_fr", ";pT [GeV];Algorithmic fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_alg_fr = TProfile("gnn_pt_alg_fr", ";pT [GeV];Algorithmic fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_alg_fr = TProfile("sv1_eta_alg_fr", ";#eta;Algorithmic fake rate",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_alg_fr = TProfile("gnn_eta_alg_fr", ";#eta;Algorithmic fake rate",20,-jet_eta_bound,jet_eta_bound)
 
    #physics efficiency/fr histograms
    sv1_pt_profile_phys_eff_c = TProfile("sv1_pt_phys_eff_c", ";pT [GeV];Physics efficiency (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_phys_eff_c = TProfile("gnn_pt_phys_eff_c", ";pT [GeV];Physics efficiency (c-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_phys_eff_c = TProfile("sv1_eta_phys_eff_c", ";#eta;Physics efficiency (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_phys_eff_c = TProfile("gnn_eta_phys_eff_c", ";#eta;Physics efficiency (c-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_pt_profile_phys_eff_b = TProfile("sv1_pt_phys_eff_b", ";pT [GeV];Physics efficiency (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_phys_eff_b = TProfile("gnn_pt_phys_eff_b", ";pT [GeV];Physics efficiency (b-jets)",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_phys_eff_b = TProfile("sv1_eta_phys_eff_b", ";#eta;Physics efficiency (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_phys_eff_b = TProfile("gnn_eta_phys_eff_b", ";#eta;Physics efficiency (b-jets)",20,-jet_eta_bound,jet_eta_bound)
    sv1_ntrk_profile_phys_eff_c = TProfile("sv1_ntrk_phys_eff_c", ";Number of tracks;Physics efficiency (c-jets)",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_phys_eff_c = TProfile("gnn_ntrk_phys_eff_c", ";Number of tracks;Physics efficiency (c-jets)",ntrk_bound+1,bin_edges_ntrk)
    sv1_ntrk_profile_phys_eff_b = TProfile("sv1_ntrk_phys_eff_b", ";Number of tracks;Physics efficiency (b-jets)",ntrk_bound+1,bin_edges_ntrk)
    gnn_ntrk_profile_phys_eff_b = TProfile("gnn_ntrk_phys_eff_b", ";Number of tracks;Physics efficiency (b-jets)",ntrk_bound+1,bin_edges_ntrk)
    sv1_pt_profile_phys_fr = TProfile("sv1_pt_phys_fr", ";pT [GeV];Physics fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_phys_fr = TProfile("gnn_pt_phys_fr", ";pT [GeV];Physics fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_phys_fr = TProfile("sv1_eta_phys_fr", ";#eta;Physics fake rate",20,-jet_eta_bound,jet_eta_bound)
    gnn_eta_profile_phys_fr = TProfile("gnn_eta_phys_fr", ";#eta;Physics fake rate",20,-jet_eta_bound,jet_eta_bound)
    
    #histograms showing track flavor label prevalence in falsely associated tracks
    gnn_trk_assoc_hist_fake = TH1D("gnn_trk_assoc_fake", ";Track flavor label;Number of tracks", len(trk_flavor_labels), 0, len(trk_flavor_labels))
    gnn_trk_assoc_hist_true = TH1D("gnn_trk_assoc_true", ";Track flavor label; Number of tracks", len(trk_flavor_labels), 0, len(trk_flavor_labels))

    #initialize matrices for overall SV predictions and plots
    gnn_cm = np.zeros((3,3), dtype=int)
    sv1_cm = np.zeros((3,3), dtype=int)
    b_tp_sv1_alg = b_fn_sv1_alg = c_tp_sv1_alg = c_fn_sv1_alg = fp_sv1_alg = tn_sv1_alg = 0
    b_tp_gnn_alg = b_fn_gnn_alg = c_tp_gnn_alg = c_fn_gnn_alg = fp_gnn_alg = tn_gnn_alg = 0
    b_tp_sv1_phys = b_fn_sv1_phys = c_tp_sv1_phys = c_fn_sv1_phys = fp_sv1_phys = tn_sv1_phys = 0
    b_tp_gnn_phys = b_fn_gnn_phys = c_tp_gnn_phys = c_fn_gnn_phys = fp_gnn_phys = tn_gnn_phys = 0

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

            gnn_vertices = find_vertices_bin(g, 'gnn', score_threshold)
            true_vertices = find_vertices_bin(g, 'truth', 0.9999)
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
            #for each true vertex, fill histograms showing number of correctly/falsely associated tracks
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

            #fill algorithmic efficiency profiles for b/c-jets
            if no_b > 0:
                b_tp_sv1_alg += jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]
                b_fn_sv1_alg += jet_sv1_cm[1,0]+jet_sv1_cm[2,0]
                b_tp_gnn_alg += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]
                b_fn_gnn_alg += jet_gnn_cm[1,0]+jet_gnn_cm[2,0]
                sv1_pt_profile_alg_eff_b.Fill(jet_pt, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_pt_profile_alg_eff_b.Fill(jet_pt, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
                sv1_eta_profile_alg_eff_b.Fill(jet_eta, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_eta_profile_alg_eff_b.Fill(jet_eta, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
                sv1_ntrk_profile_alg_eff_b.Fill(ntrk, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_ntrk_profile_alg_eff_b.Fill(ntrk, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
                sv1_lxy_profile_alg_eff_b.Fill(Lxy,float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_lxy_profile_alg_eff_b.Fill(Lxy,float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
            elif no_c > 0:
                c_tp_sv1_alg += jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]
                c_fn_sv1_alg += jet_sv1_cm[1,0]+jet_sv1_cm[2,0]
                c_tp_gnn_alg += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]
                c_fn_gnn_alg += jet_gnn_cm[1,0]+jet_gnn_cm[2,0]
                sv1_pt_profile_alg_eff_c.Fill(jet_pt, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_pt_profile_alg_eff_c.Fill(jet_pt, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
                sv1_eta_profile_alg_eff_c.Fill(jet_eta, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_eta_profile_alg_eff_c.Fill(jet_eta, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
                sv1_ntrk_profile_alg_eff_c.Fill(ntrk, float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_ntrk_profile_alg_eff_c.Fill(ntrk, float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
                sv1_lxy_profile_alg_eff_c.Fill(Lxy,float(jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]))
                gnn_lxy_profile_alg_eff_c.Fill(Lxy,float(jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]))
            else:
                fp_sv1_alg += jet_sv1_cm[0,1]+jet_sv1_cm[0,2]
                tn_sv1_alg += jet_sv1_cm[0,0]
                fp_gnn_alg += jet_gnn_cm[0,1]+jet_gnn_cm[0,2]
                tn_gnn_alg += jet_gnn_cm[0,0]

            #fill algorithmic fake rate profiless
            if (jet_sv1_cm[0,1]+jet_sv1_cm[0,2]+jet_sv1_cm[1,1]+jet_sv1_cm[1,2]+jet_sv1_cm[2,1]+jet_sv1_cm[2,2]) == 1:
                sv1_pt_profile_alg_fr.Fill(jet_pt, float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2]))
                sv1_eta_profile_alg_fr.Fill(jet_eta, float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2]))
            if (jet_gnn_cm[0,1]+jet_gnn_cm[0,2]+jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_sv1_cm[2,2]) == 1:
                gnn_pt_profile_alg_fr.Fill(jet_pt, float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2]))
                gnn_eta_profile_alg_fr.Fill(jet_eta, float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2])) 

            #fill physics efficiency profiles for b/c-jets
            if jet_flavor == 1:
                b_tp_sv1_phys += float(len(sv1_vertex) != 0)
                b_fn_sv1_phys += float(len(sv1_vertex) == 0)
                b_tp_gnn_phys += float(len(gnn_vertices) != 0)
                b_fn_gnn_phys += float(len(gnn_vertices) == 0)
                sv1_pt_profile_phys_eff_b.Fill(jet_pt, float(len(sv1_vertex) != 0))
                gnn_pt_profile_phys_eff_b.Fill(jet_pt, float(len(gnn_vertices) != 0))
                sv1_eta_profile_phys_eff_b.Fill(jet_eta, float(len(sv1_vertex) != 0))
                gnn_eta_profile_phys_eff_b.Fill(jet_eta, float(len(gnn_vertices) != 0))
                sv1_ntrk_profile_phys_eff_b.Fill(ntrk, float(len(sv1_vertex) != 0))
                gnn_ntrk_profile_phys_eff_b.Fill(ntrk, float(len(gnn_vertices) != 0))
            elif jet_flavor == 2:
                c_tp_sv1_phys += float(len(sv1_vertex) != 0)
                c_fn_sv1_phys += float(len(sv1_vertex) == 0)
                c_tp_gnn_phys += float(len(gnn_vertices) != 0)
                c_fn_gnn_phys += float(len(gnn_vertices) == 0)
                sv1_pt_profile_phys_eff_c.Fill(jet_pt, float(len(sv1_vertex) != 0))
                gnn_pt_profile_phys_eff_c.Fill(jet_pt, float(len(gnn_vertices) != 0))
                sv1_eta_profile_phys_eff_c.Fill(jet_eta, float(len(sv1_vertex) != 0))
                gnn_eta_profile_phys_eff_c.Fill(jet_eta, float(len(gnn_vertices) != 0))
                sv1_ntrk_profile_phys_eff_c.Fill(ntrk, float(len(sv1_vertex) != 0))
                gnn_ntrk_profile_phys_eff_c.Fill(ntrk, float(len(gnn_vertices) != 0))
            else:
                fp_sv1_phys += float(len(sv1_vertex) != 0)
                tn_sv1_phys += float(len(sv1_vertex) == 0)
                fp_gnn_phys += float(len(gnn_vertices) != 0)
                tn_gnn_phys += float(len(gnn_vertices) == 0)

            #fill physics fake rate histograms
            if (float(len(sv1_vertex) > 0)):
                sv1_pt_profile_phys_fr.Fill(jet_pt, float(jet_flavor == 0))
                sv1_eta_profile_phys_fr.Fill(jet_eta, float(jet_flavor == 0))
            if (float(len(gnn_vertices) > 0)):
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
        print(f'Global algorithmic b-jet efficiency: {b_tp_gnn_alg/(b_tp_gnn_alg+b_fn_gnn_alg):4} (GNN), {b_tp_sv1_alg/(b_tp_sv1_alg+b_fn_sv1_alg):4} (SV1)')
        print(f'Global algorithmic c-jet efficiency: {c_tp_gnn_alg/(c_tp_gnn_alg+c_fn_gnn_alg):4} (GNN), {c_tp_sv1_alg/(c_tp_sv1_alg+c_fn_sv1_alg):4} (SV1)')
        print(f'Global algorithmic fake rate: {fp_gnn_alg/(fp_gnn_alg+b_tp_gnn_alg+c_tp_gnn_alg):4} (GNN), {fp_sv1_alg/(fp_sv1_alg+b_tp_sv1_alg+c_tp_sv1_alg):4} (SV1)')
        print(f'Global physics b-jet efficiency: {b_tp_gnn_phys/(b_tp_gnn_phys+b_fn_gnn_phys):4} (GNN), {b_tp_sv1_phys/(b_tp_sv1_phys+b_fn_sv1_phys):4} (SV1)')
        print(f'Global physics c-jet efficiency: {c_tp_gnn_phys/(c_tp_gnn_phys+c_fn_gnn_phys):4} (GNN), {c_tp_sv1_phys/(c_tp_sv1_phys+c_fn_sv1_phys):4} (SV1)')
        print(f'Global physics fake rate: {fp_gnn_phys/(fp_gnn_phys+b_tp_gnn_phys+c_tp_gnn_phys):4} (GNN), {fp_sv1_phys/(fp_sv1_phys+b_tp_sv1_phys+c_tp_sv1_phys):4} (SV1)')

    canv1 = TCanvas("c1", "c1",200,10,900,900)
    plot_profileratio(canv1, [gnn_pt_profile_alg_eff_c], [sv1_pt_profile_alg_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_pt_c.png")
    plot_profileratio(canv1, [gnn_pt_profile_alg_eff_b], [sv1_pt_profile_alg_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_pt_b.png")
    plot_profileratio(canv1, [gnn_eta_profile_alg_eff_c], [sv1_eta_profile_alg_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_eta_c.png")
    plot_profileratio(canv1, [gnn_eta_profile_alg_eff_b], [sv1_eta_profile_alg_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_eta_b.png")
    plot_profileratio(canv1, [gnn_lxy_profile_alg_eff_b], [sv1_lxy_profile_alg_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_lxy_b.png")
    plot_profileratio(canv1, [gnn_lxy_profile_alg_eff_c], [sv1_lxy_profile_alg_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_lxy_c.png")
    plot_profileratio(canv1, [gnn_ntrk_profile_alg_eff_b], [sv1_ntrk_profile_alg_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_ntrk_b.png")
    plot_profileratio(canv1, [gnn_ntrk_profile_alg_eff_c], [sv1_ntrk_profile_alg_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_eff_ntrk_c.png")
    plot_profileratio(canv1, [gnn_pt_profile_phys_eff_c], [sv1_pt_profile_phys_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_eff_pt_c.png")
    plot_profileratio(canv1, [gnn_pt_profile_phys_eff_b], [sv1_pt_profile_phys_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_eff_pt_b.png")
    plot_profileratio(canv1, [gnn_eta_profile_phys_eff_c], [sv1_eta_profile_phys_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_eff_eta_c.png")
    plot_profileratio(canv1, [gnn_eta_profile_phys_eff_b], [sv1_eta_profile_phys_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_eff_eta_b.png")
    plot_profileratio(canv1, [gnn_ntrk_profile_phys_eff_b], [sv1_ntrk_profile_phys_eff_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_eff_ntrk_b.png")
    plot_profileratio(canv1, [gnn_ntrk_profile_phys_eff_c], [sv1_ntrk_profile_phys_eff_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_eff_ntrk_c.png")
    plot_profileratio(canv1, [gnn_pt_profile_alg_fr], [sv1_pt_profile_alg_fr], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_fr_pt.png")
    plot_profileratio(canv1, [gnn_eta_profile_alg_fr], [sv1_eta_profile_alg_fr], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_alg_fr_eta.png")
    plot_profileratio(canv1, [gnn_pt_profile_phys_fr], [sv1_pt_profile_phys_fr], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_fr_pt.png")
    plot_profileratio(canv1, [gnn_eta_profile_phys_fr], [sv1_eta_profile_phys_fr], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_phys_fr_eta.png")
    plot_profileratio(canv1, [gnn_pt_profile_corrp_c], [sv1_pt_profile_corrp_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_corrp_pt_c.png")
    plot_profileratio(canv1, [gnn_pt_profile_corrp_b], [sv1_pt_profile_corrp_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_corrp_pt_b.png")
    plot_profileratio(canv1, [gnn_eta_profile_corrp_c], [sv1_eta_profile_corrp_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_corrp_eta_c.png")
    plot_profileratio(canv1, [gnn_eta_profile_corrp_b], [sv1_eta_profile_corrp_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_corrp_eta_b.png")
    plot_profileratio(canv1, [gnn_pt_profile_fakep_c], [sv1_pt_profile_fakep_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_fakep_pt_c.png")
    plot_profileratio(canv1, [gnn_pt_profile_fakep_b], [sv1_pt_profile_fakep_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_fakep_pt_b.png")
    plot_profileratio(canv1, [gnn_eta_profile_fakep_c], [sv1_eta_profile_fakep_c], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_fakep_eta_c.png")
    plot_profileratio(canv1, [gnn_eta_profile_fakep_b], [sv1_eta_profile_fakep_b], ["GNN"], ["SV1"], cut_string, False, True, outfile_name+"_fakep_eta_b.png")

    canv2 = TCanvas("c2", "c2", 800, 600) 
    plot_bar(canv2, [gnn_trk_assoc_hist_fake, gnn_trk_assoc_hist_true], trk_flavor_labels, ["Assoc w fake", "Misassoc w true"], cut_string, False, True, outfile_name+"_trk_assoc_false.png")

    ext = ["_corrp.png", "_fakep.png"]
    hist_list_list = [hist_pc_list, hist_pf_list]
    for i in range(len(hist_list_list)):
        plot_hist(canv2, hist_list_list[i], ["GNN", "SV1"], cut_string, True, False, outfile_name+ext[i])

    #evaluate roc data
    if plot_roc:
        score_threshold_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        b_tp_gnn_alg_array = np.zeros(len(score_threshold_array))
        c_tp_gnn_alg_array = np.zeros(len(score_threshold_array))
        b_fn_gnn_alg_array = np.zeros(len(score_threshold_array))
        c_fn_gnn_alg_array = np.zeros(len(score_threshold_array))
        fp_gnn_alg_array = np.zeros(len(score_threshold_array))
        tn_gnn_alg_array = np.zeros(len(score_threshold_array))
        b_tp_gnn_phys_array = np.zeros(len(score_threshold_array))
        c_tp_gnn_phys_array = np.zeros(len(score_threshold_array))
        b_fn_gnn_phys_array = np.zeros(len(score_threshold_array))
        c_fn_gnn_phys_array = np.zeros(len(score_threshold_array))
        fp_gnn_phys_array = np.zeros(len(score_threshold_array))
        tn_gnn_phys_array = np.zeros(len(score_threshold_array))

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

                    if no_b > 0:
                        b_tp_gnn_alg_array[ist] += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]
                        b_fn_gnn_alg_array[ist] += jet_gnn_cm[1,0]+jet_gnn_cm[2,0]
                    elif no_c > 0:
                        c_tp_gnn_alg_array[ist] += jet_gnn_cm[1,1]+jet_gnn_cm[1,2]+jet_gnn_cm[2,1]+jet_gnn_cm[2,2]
                        c_fn_gnn_alg_array[ist] += jet_gnn_cm[1,0]+jet_gnn_cm[2,0]
                    else:
                        fp_gnn_alg_array[ist] += jet_gnn_cm[0,1]+jet_gnn_cm[0,2]
                        tn_gnn_alg_array[ist] += jet_sv1_cm[0,0]

                    if jet_flavor == 1:
                        b_tp_gnn_phys_array[ist] += float(len(gnn_vertices) != 0)
                        b_fn_gnn_phys_array[ist] += float(len(gnn_vertices) == 0)
                    elif jet_flavor == 2:
                        c_tp_gnn_phys_array[ist] += float(len(gnn_vertices) != 0)
                        c_fn_gnn_phys_array[ist] += float(len(gnn_vertices) == 0)
                    else:
                        fp_gnn_phys_array[ist] += float(len(gnn_vertices) != 0)
                        tn_gnn_phys_array[ist] += float(len(gnn_vertices) == 0)

        b_alg_efficiency = np.divide(b_tp_gnn_alg_array,b_tp_gnn_alg_array+b_fn_gnn_alg_array)
        c_alg_efficiency = np.divide(c_tp_gnn_alg_array,c_tp_gnn_alg_array+c_fn_gnn_alg_array)
        alg_fake_rate = np.divide(fp_gnn_alg_array,fp_gnn_alg_array+b_tp_gnn_alg_array+c_tp_gnn_alg_array)
        sv1_b_alg_efficiency = np.array(b_tp_sv1_alg/(b_tp_sv1_alg+b_fn_sv1_alg))
        sv1_c_alg_efficiency = np.array(c_tp_sv1_alg/(c_tp_sv1_alg+c_fn_sv1_alg))
        sv1_alg_fake_rate = np.array(fp_sv1_alg/(fp_sv1_alg+b_tp_sv1_alg+c_tp_sv1_alg))

        b_phys_efficiency = np.divide(b_tp_gnn_phys_array,b_tp_gnn_phys_array+b_fn_gnn_phys_array)
        c_phys_efficiency = np.divide(c_tp_gnn_phys_array,c_tp_gnn_phys_array+c_fn_gnn_phys_array)
        phys_fake_rate = np.divide(fp_gnn_phys_array,fp_gnn_phys_array+b_tp_gnn_phys_array+c_tp_gnn_phys_array)
        sv1_b_phys_efficiency = np.array(b_tp_sv1_phys/(b_tp_sv1_phys+b_fn_sv1_phys))
        sv1_c_phys_efficiency = np.array(c_tp_sv1_phys/(c_tp_sv1_phys+c_fn_sv1_phys))
        sv1_phys_fake_rate = np.array(fp_sv1_phys/(fp_sv1_phys+b_tp_sv1_phys+c_tp_sv1_phys))

        alg_roc_curve_b = TGraph(len(b_alg_efficiency), alg_fake_rate, b_alg_efficiency)
        alg_roc_curve_c = TGraph(len(c_alg_efficiency), alg_fake_rate, c_alg_efficiency)
        sv1_alg_eff_b = TGraph(sv1_b_alg_efficiency.size, sv1_alg_fake_rate, sv1_b_alg_efficiency)
        sv1_alg_eff_c = TGraph(sv1_c_alg_efficiency.size, sv1_alg_fake_rate, sv1_c_alg_efficiency)
        plot_roc_curve(canv2, alg_roc_curve_b, alg_roc_curve_c, sv1_alg_eff_b, sv1_alg_eff_c, [0.0,1.0], [0.,1.0], "algorithmic", outfile_name+"_alg_roc.png")

        phys_roc_curve_b = TGraph(len(b_phys_efficiency), phys_fake_rate, b_phys_efficiency)
        phys_roc_curve_c = TGraph(len(c_phys_efficiency), phys_fake_rate, c_phys_efficiency)
        sv1_phys_eff_b = TGraph(sv1_b_phys_efficiency.size, sv1_phys_fake_rate, sv1_b_phys_efficiency)
        sv1_phys_eff_c = TGraph(sv1_c_phys_efficiency.size, sv1_phys_fake_rate, sv1_c_phys_efficiency)
        plot_roc_curve(canv2, phys_roc_curve_b, phys_roc_curve_c, sv1_phys_eff_b, sv1_phys_eff_c, [0.0,1.0], [0.,1.0], "physics", outfile_name+"_phys_roc.png")

        #calculate area under ROC curve
        alg_roc_curve_b_auc = np.trapz(np.flip(b_alg_efficiency), x=np.flip(alg_fake_rate))
        alg_roc_curve_c_auc = np.trapz(np.flip(c_alg_efficiency), x=np.flip(alg_fake_rate))
        phys_roc_curve_b_auc = np.trapz(np.flip(b_phys_efficiency), x=np.flip(phys_fake_rate))
        phys_roc_curve_c_auc = np.trapz(np.flip(c_phys_efficiency), x=np.flip(phys_fake_rate))
        print("AuC for algorithmic b: {}".format(alg_roc_curve_b_auc))
        print("AuC for algorithmic c: {}".format(alg_roc_curve_c_auc))
        print("AuC for physics b: {}".format(phys_roc_curve_b_auc))
        print("AuC for physics c: {}".format(phys_roc_curve_c_auc))

    canv2.Clear()
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
    canv2.SaveAs(outfile_name+"_no_sv.png")
    canv2.Clear()
    del canv1


if __name__ == '__main__':
    main(sys.argv)
