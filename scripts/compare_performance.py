#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')

import dgl
import torch as th
import torch.nn as nn
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
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound
    plot_roc = options.plot_roc

    graphfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_results.bin"
    paramfile_name = infile_path+infile_name+"_params"
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber
    normfile_name = infile_path+infile_name+"_norm"

    #calculate number of features in graphs
    sample_graph = (dgl.load_graphs(graphfile_name, [0]))[0][0]
    incl_errors = incl_corr = incl_hits = False
    nnfeatures_base = sample_graph.ndata['features_base'].size()[1]
    nnfeatures = nnfeatures_base
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
        test_len = int(paramfile.readline())
    else:
        print("ERROR: Specified parameter file not found")
        return 1
    batches = int(math.ceil(test_len/batch_size))

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

    bin_edges = np.linspace(-0.05,1.05,12)

    sv1_corrp_hist = TH1D("SV1", "Fraction of tracks from true SV in predicted SV;Fraction of correct tracks;Fraction of jets",11,bin_edges)
    sv1_fakep_hist = TH1D("SV1 ", "Fraction of tracks in predicted SV not found in true SV;Fraction of correct tracks;Fraction of jets",11,bin_edges)
    gnn_corrp_hist = TH1D("GNN", "Fraction of tracks from true SV in predicted SV;Fraction of correct tracks;Fraction of jets",11,bin_edges)
    gnn_fakep_hist = TH1D("GNN ", "Fraction of tracks in predicted SV not found in true SV;Fraction of correct tracks;Fraction of jets",11,bin_edges)
	
    no_true_sv_hist_tot = TH1D("no_true_sv_tot", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges[0:7]*10)
    no_true_sv_hist_c = TH1D("no_true_sv_c", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges[0:7]*10)
    no_true_sv_hist_b = TH1D("no_true_sv_b", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges[0:7]*10)
    no_true_sv_hist_btoc = TH1D("no_true_sv_btoc", "Number of true secondary vertices per jet;Number of SV's;Number of jets",6,bin_edges[0:7]*10)

    sv1_pt_profile_eff_c = TProfile("sv1_pt_eff_c", "SV reconstruction efficiency for c jets as a function of jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_eff_c = TProfile("gnn_pt_eff_c", "SV reconstruction efficiency for c jets as a function of jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_eff_c = TProfile("sv1_eta_eff_c", "SV reconstruction efficiency for c jets as a function of jet eta;eta;Efficiency",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_eff_c = TProfile("gnn_eta_eff_c", "SV reconstruction efficiency for c jets as a function of jet eta;eta;Efficiency",20,jet_eta_bound[0],jet_eta_bound[1])
    sv1_pt_profile_eff_b = TProfile("sv1_pt_eff_b", "SV reconstruction efficiency for b jets as a function of jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_eff_b = TProfile("gnn_pt_eff_b", "SV reconstruction efficiency for b jets as a function of jet pT;pT [GeV];Efficiency",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_eff_b = TProfile("sv1_eta_eff_b", "SV reconstruction efficiency for b jets as a function of jet eta;eta;Efficiency",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_eff_b = TProfile("gnn_eta_eff_b", "SV reconstruction efficiency for b jets as a function of jet eta;eta;Efficiency",20,jet_eta_bound[0],jet_eta_bound[1])

    sv1_pt_profile_fr = TProfile("sv1_pt_fr", "SV reconstruction fake rate as a function of jet pT;pT [GeV];Fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fr = TProfile("gnn_pt_fr", "SV reconstruction fake rate as a function of jet pT;pT [GeV];Fake rate",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fr = TProfile("sv1_eta_fr", "SV reconstruction fake rate as a function of jet eta;eta;Fake rate",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_fr = TProfile("gnn_eta_fr", "SV reconstruction fake rate as a function of jet eta;eta;Fake rate",20,jet_eta_bound[0],jet_eta_bound[1])
   
    sv1_pt_profile_corrp_c = TProfile("sv1_pt_corrp_c", "Fraction of tracks from true SV in predicted SV for c jets as a function of jet pT;pT [GeV];Fraction of correct tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_corrp_c = TProfile("gnn_pt_corrp_c", "Fraction of tracks from true SV in predicted SV for c jets as a function of jet pT;pT [GeV];Fraction of correct tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_corrp_c = TProfile("sv1_eta_corrp_c", "Fraction of tracks from true SV in predicted SV for c jets as a function of jet eta;eta;Fraction of correct tracks",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_corrp_c = TProfile("gnn_eta_corrp_c", "Fraction of tracks from true SV in predicted SV for c jets as a function of jet eta;eta;Fraction of correct tracks",20,jet_eta_bound[0],jet_eta_bound[1])
    sv1_pt_profile_corrp_b = TProfile("sv1_pt_corrp_b", "Fraction of tracks from true SV in predicted SV for b jets as a function of jet pT;pT [GeV];Fraction of correct tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_corrp_b = TProfile("gnn_pt_corrp_b", "Fraction of tracks from true SV in predicted SV for b jets as a function of jet pT;pT [GeV];Fraction of correct tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_corrp_b = TProfile("sv1_eta_corrp_b", "Fraction of tracks from true SV in predicted SV for b jets as a function of jet eta;eta;Fraction of correct tracks",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_corrp_b = TProfile("gnn_eta_corrp_b", "Fraction of tracks from true SV in predicted SV for b jets as a function of jet eta;eta;Fraction of correct tracks",20,jet_eta_bound[0],jet_eta_bound[1])

    sv1_pt_profile_fakep_c = TProfile("sv1_pt_fakep_c", "Fraction of tracks in predicted SV not found in true SV for c jets as a function of jet pT;pT [GeV];Fraction of false tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fakep_c = TProfile("gnn_pt_fakep_c", "Fraction of tracks in predicted SV not found in true SV for c jets as a function of jet pT;pT [GeV];Fraction of false tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fakep_c = TProfile("sv1_eta_fakep_c", "Fraction of tracks in predicted SV not found in true SV for c jets as a function of jet eta;eta;Fraction of false tracks",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_fakep_c = TProfile("gnn_eta_fakep_c", "Fraction of tracks in predicted SV not found in true SV for c jets as a function of jet eta;eta;Fraction of false tracks",20,jet_eta_bound[0],jet_eta_bound[1])
    sv1_pt_profile_fakep_b = TProfile("sv1_pt_fakep_b", "Fraction of tracks in predicted SV not found in true SV for b jets as a function of jet pT;pT [GeV];Fraction of false tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    gnn_pt_profile_fakep_b = TProfile("gnn_pt_fakep_b", "Fraction of tracks in predicted SV not found in true SV for b jets as a function of jet pT;pT [GeV];Fraction of false tracks",20,jet_pt_bound[0],jet_pt_bound[1])
    sv1_eta_profile_fakep_b = TProfile("sv1_eta_fakep_b", "Fraction of tracks in predicted SV not found in true SV for b jets as a function of jet eta;eta;Fraction of false tracks",20,jet_eta_bound[0],jet_eta_bound[1])
    gnn_eta_profile_fakep_b = TProfile("gnn_eta_fakep_b", "Fraction of tracks in predicted SV not found in true SV for b jets as a function of jet eta;eta;Fraction of false tracks",20,jet_eta_bound[0],jet_eta_bound[1])

    hist_pc_list = [sv1_corrp_hist, gnn_corrp_hist]
    hist_pf_list = [sv1_fakep_hist, gnn_fakep_hist]

    #initialize matrices for overall SV predictions and plots
    gnn_cm = np.zeros((3,3), dtype=int)
    sv1_cm = np.zeros((3,3), dtype=int)
    b_tot = b_found_gnn = b_found_sv1 = c_tot = c_found_gnn = c_found_sv1 = no_sv_tot = fake_found_gnn = fake_found_sv1 = tot_pred_gnn = tot_pred_sv1 = 0

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

            gnn_vertices = find_vertices_bin(g, 'gnn', score_threshold)
            true_vertices = find_vertices_bin(g, 'truth', score_threshold)
            sv1_vertex = np.argwhere(g.ndata['reco_labels'].cpu().numpy().astype(int)[:,1]).flatten()
            if sv1_vertex.size > 0: sv1_vertex = [sv1_vertex]

            jet_gnn_cm, gnn_vertex_metrics, gnn_vertex_assoc = compare_vertices(true_vertices, gnn_vertices)
            jet_sv1_cm, sv1_vertex_metrics, sv1_vertex_assoc = compare_vertices(true_vertices, sv1_vertex)
            gnn_cm += jet_gnn_cm
            sv1_cm += jet_sv1_cm

            jet_pt = g.ndata['features_base'][0,5]
            jet_eta = g.ndata['features_base'][0,6]
            if norm:
                jet_pt = jet_pt*jet_pt_std+jet_pt_mean
                jet_eta = jet_eta*jet_eta_std+jet_eta_mean

            no_true_sv_hist_tot.Fill(len(true_vertices))

            no_b = no_c = no_btoc = 0
            for i in range(len(true_vertices)):
                true_vertex = true_vertices[i]
                edge_id = g.edge_id(true_vertex[0],true_vertex[1])
                vertex_flavor = g.edata['mult_labels'][edge_id]
                if vertex_flavor == 1:
                    no_b += 1
                elif vertex_flavor == 2:
                    no_c += 1
                elif vertex_flavor == 3:
                    no_btoc += 1

                if gnn_vertex_assoc[i] >= 0:
                    index = gnn_vertex_assoc[i]
                    if vertex_flavor == 2:
                        gnn_pt_profile_corrp_c.Fill(jet_pt, gnn_vertex_metrics[index][0])
                        gnn_pt_profile_fakep_c.Fill(jet_pt, gnn_vertex_metrics[index][1])
                        gnn_eta_profile_corrp_c.Fill(jet_eta, gnn_vertex_metrics[index][0])
                        gnn_eta_profile_fakep_c.Fill(jet_eta, gnn_vertex_metrics[index][1]) 
                    elif vertex_flavor == 1 or vertex_flavor == 3:
                        gnn_pt_profile_corrp_b.Fill(jet_pt, gnn_vertex_metrics[index][0])
                        gnn_pt_profile_fakep_b.Fill(jet_pt, gnn_vertex_metrics[index][1])
                        gnn_eta_profile_corrp_b.Fill(jet_eta, gnn_vertex_metrics[index][0])
                        gnn_eta_profile_fakep_b.Fill(jet_eta, gnn_vertex_metrics[index][1]) 

                if sv1_vertex_assoc[i] >= 0:
                    index = sv1_vertex_assoc[i]
                    if vertex_flavor == 2:
                        sv1_pt_profile_corrp_c.Fill(jet_pt, sv1_vertex_metrics[index][0])
                        sv1_pt_profile_fakep_c.Fill(jet_pt, sv1_vertex_metrics[index][1])
                        sv1_eta_profile_corrp_c.Fill(jet_eta, sv1_vertex_metrics[index][0])
                        sv1_eta_profile_fakep_c.Fill(jet_eta, sv1_vertex_metrics[index][1]) 
                    elif vertex_flavor == 1 or vertex_flavor == 3:
                        sv1_pt_profile_corrp_b.Fill(jet_pt, sv1_vertex_metrics[index][0])
                        sv1_pt_profile_fakep_b.Fill(jet_pt, sv1_vertex_metrics[index][1])
                        sv1_eta_profile_corrp_b.Fill(jet_eta, sv1_vertex_metrics[index][0])
                        sv1_eta_profile_fakep_b.Fill(jet_eta, sv1_vertex_metrics[index][1]) 

            if no_b > 0: no_true_sv_hist_b.Fill(no_b)
            if no_c > 0: no_true_sv_hist_c.Fill(no_c)
            if no_btoc > 0: no_true_sv_hist_btoc.Fill(no_btoc)

            #fill histograms for b SVs
            if no_b > 0 or no_btoc > 0:
                b_tot += 1
                b_found_sv1 += jet_sv1_cm[1,1]+jet_sv1_cm[2,2]
                b_found_gnn += jet_gnn_cm[1,1]+jet_gnn_cm[2,2]
                sv1_pt_profile_eff_b.Fill(jet_pt, float(jet_sv1_cm[1,1]+jet_sv1_cm[2,2]))
                gnn_pt_profile_eff_b.Fill(jet_pt, float(jet_gnn_cm[1,1]+jet_gnn_cm[2,2]))
                sv1_eta_profile_eff_b.Fill(jet_eta, float(jet_sv1_cm[1,1]+jet_sv1_cm[2,2]))
                gnn_eta_profile_eff_b .Fill(jet_eta, float(jet_gnn_cm[1,1]+jet_gnn_cm[2,2]))

            #fill histograms for c SVs
            if no_c > 0 and no_b == 0 and no_btoc == 0:
                c_tot += 1
                c_found_sv1 += jet_sv1_cm[1,1]+jet_sv1_cm[2,2]
                c_found_gnn += jet_gnn_cm[1,1]+jet_gnn_cm[2,2]
                sv1_pt_profile_eff_c.Fill(jet_pt, float(jet_sv1_cm[1,1]+jet_sv1_cm[2,2]))
                gnn_pt_profile_eff_c.Fill(jet_pt, float(jet_gnn_cm[1,1]+jet_gnn_cm[2,2]))
                sv1_eta_profile_eff_c.Fill(jet_eta, float(jet_sv1_cm[1,1]+jet_sv1_cm[2,2]))
                gnn_eta_profile_eff_c .Fill(jet_eta, float(jet_gnn_cm[1,1]+jet_gnn_cm[2,2]))

            if no_c == 0 and no_b == 0 and no_btoc == 0:
                no_sv_tot += 1
                fake_found_sv1 += (1-jet_sv1_cm[0,0])
                fake_found_gnn += (1-jet_gnn_cm[0,0])

            if np.sum(jet_sv1_cm[:,1]+jet_sv1_cm[:,2]) > 0:
                tot_pred_sv1 += 1
                sv1_pt_profile_fr.Fill(jet_pt, float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2]))
                sv1_eta_profile_fr.Fill(jet_eta, float(jet_sv1_cm[0,1]+jet_sv1_cm[0,2]))

            if np.sum(jet_gnn_cm[:,1]+jet_gnn_cm[:,2]) > 0:
                tot_pred_gnn += 1
                gnn_pt_profile_fr.Fill(jet_pt, float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2]))
                gnn_eta_profile_fr .Fill(jet_eta, float(jet_gnn_cm[0,1]+jet_gnn_cm[0,2])) 

            for vertex_metric in gnn_vertex_metrics:
                hist_pc_list[0].Fill(vertex_metric[0])
                hist_pf_list[0].Fill(vertex_metric[1])

            for vertex_metric in sv1_vertex_metrics:
                hist_pc_list[1].Fill(vertex_metric[0])
                hist_pf_list[1].Fill(vertex_metric[1])

    print('Secondary Vertex prediction results per jet (GNN/SV1):')
    print('             ||  Pred no SV   |   Pred 1 SV   |   Pred >1 SV  |')
    print('---------------------------------------------------------------')
    print(f'Actual no SV || {gnn_cm[0,0]:5} /{sv1_cm[0,0]:6} | {gnn_cm[0,1]:5} /{sv1_cm[0,1]:6} | {gnn_cm[0,2]:5} /{sv1_cm[0,2]:6} |')
    print(f'Actual 1 SV  || {gnn_cm[1,0]:5} /{sv1_cm[1,0]:6} | {gnn_cm[1,1]:5} /{sv1_cm[1,1]:6} | {gnn_cm[1,2]:5} /{sv1_cm[1,2]:6} |')
    print(f'Actual >1 SV || {gnn_cm[2,0]:5} /{sv1_cm[2,0]:6} | {gnn_cm[2,1]:5} /{sv1_cm[2,1]:6} | {gnn_cm[2,2]:5} /{sv1_cm[2,2]:6} |')
    print('---------------------------------------------------------------')
    print(f'Edge score threshold: {score_threshold}')
    print(f'Global b-jet vertex efficiency: {b_found_gnn/b_tot:4} (GNN), {b_found_sv1/b_tot:4} (SV1)')
    print(f'Global c-jet vertex efficiency: {c_found_gnn/c_tot:4} (GNN), {c_found_sv1/c_tot:4} (SV1)')
    print(f'Global vertex fake rate: {fake_found_gnn/tot_pred_gnn:4} (GNN), {fake_found_sv1/tot_pred_sv1:4} (SV1)')

    plot_profile([sv1_pt_profile_eff_c, gnn_pt_profile_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_eff_pt_c.png")
    plot_profile([sv1_pt_profile_eff_b, gnn_pt_profile_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_eff_pt_b.png")
    plot_profile([sv1_eta_profile_eff_c, gnn_eta_profile_eff_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_eff_eta_c.png")
    plot_profile([sv1_eta_profile_eff_b, gnn_eta_profile_eff_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_eff_eta_b.png")
    plot_profile([sv1_pt_profile_fr, gnn_pt_profile_fr], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fr_pt.png")
    plot_profile([sv1_eta_profile_fr, gnn_eta_profile_fr], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fr_eta.png")
    plot_profile([sv1_pt_profile_corrp_c, gnn_pt_profile_corrp_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_pt_c.png")
    plot_profile([sv1_pt_profile_corrp_b, gnn_pt_profile_corrp_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_pt_b.png")
    plot_profile([sv1_eta_profile_corrp_c, gnn_eta_profile_corrp_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_eta_c.png")
    plot_profile([sv1_eta_profile_corrp_b, gnn_eta_profile_corrp_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_corrp_eta_b.png")
    plot_profile([sv1_pt_profile_fakep_c, gnn_pt_profile_fakep_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_pt_c.png")
    plot_profile([sv1_pt_profile_fakep_b, gnn_pt_profile_fakep_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_pt_b.png")
    plot_profile([sv1_eta_profile_fakep_c, gnn_eta_profile_fakep_c], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_eta_c.png")
    plot_profile([sv1_eta_profile_fakep_b, gnn_eta_profile_fakep_b], ["SV1", "GNN"], [0.0, 1.4], False, outfile_name+"_fakep_eta_b.png")

    ext = ["_corrp.png", "_fakep.png"]
    hist_list_list = [hist_pc_list, hist_pf_list]
    for i in range(len(hist_list_list)):
        plot_metric_hist(hist_list_list[i], [0.0,1.4], outfile_name+ext[i])

    canv1 = TCanvas("c1", "c1", 800, 600)

    #evaluate roc data
    if plot_roc:
        score_threshold_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        c_efficiency = np.zeros(len(score_threshold_array))
        b_efficiency = np.zeros(len(score_threshold_array))
        fake_rate_num = np.zeros(len(score_threshold_array))
        fake_rate_denom = np.zeros(len(score_threshold_array))

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
                    gnn_vertices = find_vertices_bin(g, 'gnn', score_threshold)
                    true_vertices = find_vertices_bin(g, 'truth', score_threshold)
                    jet_gnn_cm, gnn_vertex_metrics, gnn_vertex_assoc = compare_vertices(true_vertices, gnn_vertices)

                    no_b = no_c = no_btoc = 0
                    for i in range(len(true_vertices)):
                        true_vertex = true_vertices[i]
                        edge_id = g.edge_id(true_vertex[0],true_vertex[1])
                        vertex_flavor = g.edata['mult_labels'][edge_id]
                        if vertex_flavor == 1:
                            no_b += 1
                        elif vertex_flavor == 2:
                            no_c += 1
                        elif vertex_flavor == 3:
                            no_btoc += 1

                    #fill histograms for b SVs
                    if no_b > 0 or no_btoc > 0:
                        b_efficiency[ist] += jet_gnn_cm[1,1]+jet_gnn_cm[2,2]

                    #fill histograms for c SVs
                    if no_c > 0 and no_b == 0 and no_btoc == 0:
                        c_efficiency[ist] += jet_gnn_cm[1,1]+jet_gnn_cm[2,2]

                    if no_c == 0 and no_b == 0 and no_btoc == 0:
                        fake_rate_num[ist] += (1-jet_gnn_cm[0,0])

                    fake_rate_denom[ist] += (1-np.sum(jet_gnn_cm[:,0]))

        b_efficiency = b_efficiency/b_tot
        c_efficiency = c_efficiency/c_tot
        fake_rate = np.divide(fake_rate_num,fake_rate_denom)
        sv1_b_efficiency = np.array([b_found_sv1/b_tot])
        sv1_c_efficiency = np.array([c_found_sv1/c_tot])
        sv1_fake_rate = np.array([fake_found_sv1/tot_pred_sv1])

        mg = TMultiGraph()
        roc_curve_b = TGraph(len(score_threshold_array), b_efficiency, fake_rate)
        roc_curve_c = TGraph(len(score_threshold_array), c_efficiency, fake_rate)
        sv1_eff_b = TGraph(1, sv1_b_efficiency, sv1_fake_rate)
        sv1_eff_c = TGraph(1, sv1_c_efficiency, sv1_fake_rate)
        roc_curve_b.SetLineColor(1)
        roc_curve_c.SetLineColor(4)
        sv1_eff_b.SetLineColor(1)
        sv1_eff_c.SetLineColor(4)
        roc_curve_b.SetMarkerColor(1)
        roc_curve_c.SetMarkerColor(4)
        sv1_eff_b.SetMarkerColor(1)
        sv1_eff_c.SetMarkerColor(4)
        roc_curve_b.SetMarkerStyle(20)
        roc_curve_c.SetMarkerStyle(20)
        sv1_eff_b.SetMarkerStyle(22)
        sv1_eff_c.SetMarkerStyle(22)
        mg.Add(roc_curve_b)
        mg.Add(roc_curve_c)
        mg.Add(sv1_eff_b)
        mg.Add(sv1_eff_c)
        mg.SetTitle("; Efficiency; Fake rate")
        mg.Draw("ALP")
        mg.GetXaxis().SetLimits(0.7,1.)
        mg.SetMinimum(0.)
        mg.SetMaximum(0.3)
        legend = TLegend(0.15,0.88-0.08*4,0.3,0.88,'','NDC')
        legend.AddEntry(roc_curve_b, "b-jets (GNN)","lp")
        legend.AddEntry(roc_curve_c, "c-jets (GNN)","lp")
        legend.AddEntry(sv1_eff_b, "b-jets (SV1)","p")
        legend.AddEntry(sv1_eff_c, "c-jets (SV1)","p")
        legend.SetTextSize(0.025)
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)
        legend.Draw("SAME")
        canv1.SaveAs(outfile_name+"_roc.png")
        canv1.Clear()

    gStyle.SetOptStat(0)
    gPad.SetLogy()
    hist_list = [no_true_sv_hist_tot, no_true_sv_hist_b, no_true_sv_hist_c, no_true_sv_hist_btoc]
    colorlist = [1,4,2,8]
    labellist = ['total', 'bH', 'prompt cH', 'bH->cH']
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