#!/usr/bin/env python

####################################### plot_results.py #######################################
# PURPOSE: make plots of GNN results
# EDIT TO: add more plots, update graph structure (if modified in create_graphs)
# -------------------------------------------Summary-------------------------------------------
# This script generates plots showing results of GNN predictions. This includes both edge score
# and class recall plots in addition to plots comparing "badly reconstructed" GNN jets with the
# overall sample. The definition of what constitutes a "badly reconstructed" jet is not set in
# this script, but rather in GNN_eval. Most plots take the form of normalized histogram
# differences to make relative excesses of certain types of events that are "badly
# reconstructed" easier to spot. The plots also generally split tracks up based on their given
# track label (as in plot_data). This script can be edited to add additional plots if needed.
###############################################################################################


import dgl
import torch as th
import os,sys,math,glob,ROOT
from ROOT import TH1D, TCanvas
import numpy as np
import argparse

from plot_functions import *
from GNN_eval import *

#set ATLAS style for plots
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasStyle.C")
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasLabels.C")
from ROOT import SetAtlasStyle


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
    track_pt_bound = options.track_pt_bound
    track_d0_bound = options.track_d0_bound
    track_z0_bound = options.track_z0_bound
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound
    ntrk_bound = options.ntrk_bound
    bin_threshold = options.bin_threshold
    mult_threshold = options.mult_threshold
    cut_string = options.cut_string

    graphfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_results.bin"
    paramfile_name = infile_path+infile_name+"_params"
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber

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
    
    #check if multi-class classification was performed
    if sample_graph.edata['pred'][0,:].numpy().size > 1:
        multi_class = True
    else:
        multi_class = False

    hist_trk_pt_b_tot = TH1D("trk_pt_b_tot", ";pT [GeV];Normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_c_tot = TH1D("trk_pt_c_tot", ";pT [GeV];Normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_btoc_tot = TH1D("trk_pt_btoc_tot", ";pT [GeV];Normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_b_bad = TH1D("trk_pt_b_bad", ";pT [GeV];Normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_c_bad = TH1D("trk_pt_c_bad", ";pT [GeV];Normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_btoc_bad = TH1D("trk_pt_btoc_bad", ";pT [GeV];Normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    
    hist_trk_theta_b_tot = TH1D("trk_theta_b_tot", ";#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_c_tot = TH1D("trk_theta_c_tot", ";#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_btoc_tot = TH1D("trk_theta_btoc_tot", ";#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_b_bad = TH1D("trk_theta_b_bad", ";#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_c_bad = TH1D("trk_theta_c_bad", ";#theta;Normalized entries", 20, 0, math.pi)
    hist_trk_theta_btoc_bad = TH1D("trk_theta_btoc_bad", ";#theta;Normalized entries", 20, 0, math.pi) 

    hist_trk_phi_b_tot = TH1D("trk_phi_b_tot", ";#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c_tot = TH1D("trk_phi_c_tot", ";#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc_tot = TH1D("trk_phi_btoc_tot", ";#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_b_bad = TH1D("trk_phi_b_bad", ";#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c_bad = TH1D("trk_phi_c_bad", ";#phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc_bad = TH1D("trk_phi_btoc_bad", ";#phi;Normalized entries", 20, -math.pi, math.pi)

    hist_trk_z0_b_tot = TH1D("trk_z0_b_tot", ";z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_c_tot = TH1D("trk_z0_c_tot", ";z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_btoc_tot = TH1D("trk_z0_btoc_tot", ";z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_b_bad = TH1D("trk_z0_b_bad", ";z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_c_bad = TH1D("trk_z0_c_bad", ";z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)
    hist_trk_z0_btoc_bad = TH1D("trk_z0_btoc_bad", ";z0 [mm];Normalized entries", 20, -track_z0_bound, track_z0_bound)

    hist_trk_d0_b_tot = TH1D("trk_d0_b_tot", ";d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_c_tot = TH1D("trk_d0_c_tot", ";d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_btoc_tot = TH1D("trk_d0_btoc_tot", ";d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_b_bad = TH1D("trk_d0_b_bad", ";d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_c_bad = TH1D("trk_d0_c_bad", ";d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)
    hist_trk_d0_btoc_bad = TH1D("trk_d0_btoc_bad", ";d0 [mm];Normalized entries", 20, -track_d0_bound, track_d0_bound)

    bin_edges = np.linspace(-0.5,ntrk_bound+0.5,ntrk_bound+2)
    hist_no_trk_jet_b_tot = TH1D("no_trk_jet_b_tot", ";Number of tracks (all jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_c_tot = TH1D("no_trk_jet_c_tot", ";Number of tracks (all jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_btoc_tot = TH1D("no_trk_jet_btoc_tot", ";Number of tracks (all jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_nohf_tot = TH1D("no_trk_jet_nohf_tot", ";Number of tracks (all jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_nm_tot = TH1D("no_trk_jet_nm_tot", ";Number of tracks (all jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_b_bad = TH1D("no_trk_jet_b_bad", ";Number of tracks (badly reconstructed jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_c_bad = TH1D("no_trk_jet_c_bad", ";Number of tracks (badly reconstructed jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_btoc_bad = TH1D("no_trk_jet_btoc_bad", ";Number of tracks (badly reconstructed jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_nohf_bad = TH1D("no_trk_jet_nohf_bad", ";Number of tracks (badly reconstructed jets);Normalized entries", ntrk_bound+1, bin_edges)
    hist_no_trk_jet_nm_bad = TH1D("no_trk_jet_nm_bad", ";Number of tracks (badly reconstructed jets);Normalized entries", ntrk_bound+1, bin_edges)

    hist_frac_trk_b_tot = TH1D("frac_trk_b_tot", ";Fraction of tracks per jet (all jets);Entries", 10, 0, 1)
    hist_frac_trk_c_tot = TH1D("frac_trk_c_tot", ";Fraction of tracks per jet (all jets);Entries", 10, 0, 1)
    hist_frac_trk_nohf_tot = TH1D("frac_trk_nohf_tot", ";Fraction of tracks per jet (all jets);Entries", 10, 0, 1)
    hist_frac_trk_nm_tot = TH1D("frac_trk_nm_tot", ";Fraction of tracks per jet (all jets);Entries", 10, 0, 1)
    hist_frac_trk_btoc_tot = TH1D("frac_trk_btoc_tot", ";Fraction of tracks per jet (all jets);Entries", 10, 0, 1)
    hist_frac_trk_b_bad = TH1D("frac_trk_b_bad", ";Fraction of tracks per jet (badly reconstructed jets);Entries", 10, 0, 1)
    hist_frac_trk_c_bad = TH1D("frac_trk_c_bad", ";Fraction of tracks per jet (badly reconstructed jets);Entries", 10, 0, 1)
    hist_frac_trk_nohf_bad = TH1D("frac_trk_nohf_bad", ";Fraction of tracks per jet (badly reconstructed jets);Entries", 10, 0, 1)
    hist_frac_trk_nm_bad = TH1D("frac_trk_nm_bad", ";Fraction of tracks per jet (badly reconstructed jets);Entries", 10, 0, 1)
    hist_frac_trk_btoc_bad = TH1D("frac_trk_btoc_bad", ";Fraction of tracks per jet (badly reconstructed jets);Entries", 10, 0, 1)

    #edge score and recall histograms
    bin_edges = np.linspace(-0.05,1.05,12)
    if not multi_class:
        pos_r_hist = TH1D("TPR", ";Score;Fraction of jets",11,bin_edges) #1.001 is the upper bound so this is inclusive of 1
        neg_r_hist = TH1D("TNR", ";Score;Fraction of jets",11,bin_edges)
        edge_score_hist = TH1D("", ";Score;Fraction of jets",11,bin_edges)
        hist_r_list = [neg_r_hist, pos_r_hist] #TNR, TPR
        r_labels = ["TNR", "TPR"]
        hist_s_list = [edge_score_hist]
        s_labels = ["Edge score"]
    else:
        neg_r_hist = TH1D("Class 0 recall", ";Score;Fraction of jets",11,bin_edges)
        b_r_hist = TH1D("Class 1 recall", ";Score;Fraction of jets",11,bin_edges)
        c_r_hist = TH1D("Class 2 recall", ";Score;Fraction of jets",11,bin_edges)
        neg_score_hist = TH1D("Class 0 scores", ";Score;Fraction of jets",11,bin_edges)
        b_score_hist = TH1D("Class 1 scores", ";Score;Fraction of jets",11,bin_edges)
        c_score_hist = TH1D("Class 2 scores", ";Score;Fraction of jets",11,bin_edges)
        hist_r_list = [neg_r_hist, b_r_hist, c_r_hist]
        r_labels = ["negative recall", "b recall", "c recall"]
        hist_s_list = [neg_score_hist, b_score_hist, c_score_hist]
        s_labels = ["negative score", "b score", "c score"]

    #initialize overall bad events matrix
    bad_events = np.empty((0,3), dtype=int)

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
            features = g.ndata['features_base'].numpy()
            pred = g.edata['pred'].numpy()
            track_labels = g.ndata['track_info'].numpy()[:,0]

            #fill edge score histograms
            if not multi_class:
                for i in range(pred.shape[0]):
                    edge_score_hist.Fill(pred[i,0])
            else:
                for i in range(pred.shape[0]):
                    neg_score_hist.Fill(pred[i,0])
                    b_score_hist.Fill(pred[i,1])
                    c_score_hist.Fill(pred[i,2])

            #evaluate bad jets
            if is_bad_jet(g, hist_r_list, multi_class, bin_threshold, mult_threshold):
                bad_jet = True
            else:
                bad_jet = False

            ntrk = len(features[:,0])
            b_trk = c_trk = btoc_trk = nm_trk = nohf_trk = 0
            for i in range(ntrk):
                if track_labels[i] == 0:
                    nm_trk += 1
                elif track_labels[i] == 1:
                    b_trk += 1
                    hist_trk_pt_b_tot.Fill(abs(1/features[i,0]))
                    hist_trk_theta_b_tot.Fill(features[i,1])
                    hist_trk_phi_b_tot.Fill(features[i,2])
                    hist_trk_d0_b_tot.Fill(features[i,3])
                    hist_trk_z0_b_tot.Fill(features[i,4])
                    if bad_jet:
                        hist_trk_pt_b_bad.Fill(abs(1/features[i,0]))
                        hist_trk_theta_b_bad.Fill(features[i,1])
                        hist_trk_phi_b_bad.Fill(features[i,2])
                        hist_trk_d0_b_bad.Fill(features[i,3])
                        hist_trk_z0_b_bad.Fill(features[i,4])
                elif track_labels[i] == 2:
                    c_trk += 1
                    hist_trk_pt_c_tot.Fill(abs(1/features[i,0]))
                    hist_trk_theta_c_tot.Fill(features[i,1])
                    hist_trk_phi_c_tot.Fill(features[i,2])
                    hist_trk_d0_c_tot.Fill(features[i,3])
                    hist_trk_z0_c_tot.Fill(features[i,4])
                    if bad_jet:
                        hist_trk_pt_c_bad.Fill(abs(1/features[i,0]))
                        hist_trk_theta_c_bad.Fill(features[i,1])
                        hist_trk_phi_c_bad.Fill(features[i,2])
                        hist_trk_d0_c_bad.Fill(features[i,3])
                        hist_trk_z0_c_bad.Fill(features[i,4])
                elif track_labels[i] == 3:
                    btoc_trk += 1
                    hist_trk_pt_btoc_tot.Fill(abs(1/features[i,0]))
                    hist_trk_theta_btoc_tot.Fill(features[i,1])
                    hist_trk_phi_btoc_tot.Fill(features[i,2])
                    hist_trk_d0_btoc_tot.Fill(features[i,3])
                    hist_trk_z0_btoc_tot.Fill(features[i,4])
                    if bad_jet:
                        hist_trk_pt_btoc_bad.Fill(abs(1/features[i,0]))
                        hist_trk_theta_btoc_bad.Fill(features[i,1])
                        hist_trk_phi_btoc_bad.Fill(features[i,2])
                        hist_trk_d0_btoc_bad.Fill(features[i,3])
                        hist_trk_z0_btoc_bad.Fill(features[i,4])
                else:
                    nohf_trk += 1

            hist_no_trk_jet_b_tot.Fill(b_trk)
            hist_no_trk_jet_c_tot.Fill(c_trk)
            hist_no_trk_jet_btoc_tot.Fill(btoc_trk)
            hist_no_trk_jet_nm_tot.Fill(nm_trk)
            hist_no_trk_jet_nohf_tot.Fill(nohf_trk)
            hist_frac_trk_b_tot.Fill(b_trk/ntrk)
            hist_frac_trk_c_tot.Fill(c_trk/ntrk)
            hist_frac_trk_btoc_tot.Fill(btoc_trk/ntrk)
            hist_frac_trk_nm_tot.Fill(nm_trk/ntrk)
            hist_frac_trk_nohf_tot.Fill(nohf_trk/ntrk)
            if bad_jet:
                hist_no_trk_jet_b_bad.Fill(b_trk)
                hist_no_trk_jet_c_bad.Fill(c_trk)
                hist_no_trk_jet_btoc_bad.Fill(btoc_trk)
                hist_no_trk_jet_nm_bad.Fill(nm_trk)
                hist_no_trk_jet_nohf_bad.Fill(nohf_trk) 
                hist_frac_trk_b_bad.Fill(b_trk/ntrk)
                hist_frac_trk_c_bad.Fill(c_trk/ntrk)
                hist_frac_trk_btoc_bad.Fill(btoc_trk/ntrk)
                hist_frac_trk_nm_bad.Fill(nm_trk/ntrk)
                hist_frac_trk_nohf_bad.Fill(nohf_trk/ntrk)

    base_filename = outfile_path+runnumber+"/"+infile_name+"_"+runnumber

    canv1 = TCanvas("c1", "c1", 800, 600)
    ext = ["_recall.png", "_score.png"]
    hist_list_list = [hist_r_list, hist_s_list]
    label_list_list = [r_labels, s_labels]
    for i in range(len(hist_list_list)):
        plot_hist(canv1, hist_list_list[i], label_list_list[i], cut_string, True, True, base_filename+ext[i])
    plot_hist(canv1, [hist_no_trk_jet_b_tot, hist_no_trk_jet_c_tot, hist_no_trk_jet_btoc_tot, hist_no_trk_jet_nohf_tot, hist_no_trk_jet_nm_tot], ["bH", "prompt cH", "bH->cH", "no HF", "no match"], cut_string, True, True, base_filename+"_no_trk_tot.png")
    plot_hist(canv1, [hist_frac_trk_b_tot, hist_frac_trk_c_tot, hist_frac_trk_btoc_tot, hist_frac_trk_nohf_tot, hist_frac_trk_nm_tot], ["bH", "prompt cH", "bH->cH", "no HF", "no match"], cut_string, True, True, base_filename+"_frac_trk_tot.png")
    plot_hist(canv1, [hist_no_trk_jet_b_bad, hist_no_trk_jet_c_bad, hist_no_trk_jet_btoc_bad, hist_no_trk_jet_nohf_bad, hist_no_trk_jet_nm_bad], ["bH", "prompt cH", "bH->cH", "no HF", "no match"], cut_string, True, True, base_filename+"_no_trk_bad.png")
    plot_hist(canv1, [hist_frac_trk_b_bad, hist_frac_trk_c_bad, hist_frac_trk_btoc_bad, hist_frac_trk_nohf_bad, hist_frac_trk_nm_bad], ["bH", "prompt cH", "bH->cH", "no HF", "no match"], cut_string, True, True, base_filename+"_frac_trk_bad.png") 
    
    canv2 = TCanvas("c2", "c2",200,10,900,900)
    plot_histratio(canv2, [hist_trk_pt_b_bad, hist_trk_pt_c_bad, hist_trk_pt_btoc_bad], [hist_trk_pt_b_tot, hist_trk_pt_c_tot, hist_trk_pt_btoc_tot], ["bH", "prompt cH", "bH->cH"], cut_string, True, base_filename+"_trk_pt_comp.png")
    plot_histratio(canv2, [hist_trk_theta_b_bad, hist_trk_theta_c_bad, hist_trk_theta_btoc_bad], [hist_trk_theta_b_tot, hist_trk_theta_c_tot, hist_trk_theta_btoc_tot], ["bH", "prompt cH", "bH->cH"], cut_string, True, base_filename+"_trk_theta_comp.png")
    plot_histratio(canv2, [hist_trk_phi_b_bad, hist_trk_phi_c_bad, hist_trk_phi_btoc_bad], [hist_trk_phi_b_tot, hist_trk_phi_c_tot, hist_trk_phi_btoc_tot], ["bH", "prompt cH", "bH->cH"], cut_string, True, base_filename+"_trk_phi_comp.png")
    plot_histratio(canv2, [hist_trk_d0_b_bad, hist_trk_d0_c_bad, hist_trk_d0_btoc_bad], [hist_trk_d0_b_tot, hist_trk_d0_c_tot, hist_trk_d0_btoc_tot], ["bH", "prompt cH", "bH->cH"], cut_string, True, base_filename+"_trk_d0_comp.png")
    plot_histratio(canv2, [hist_trk_z0_b_bad, hist_trk_z0_c_bad, hist_trk_z0_btoc_bad], [hist_trk_z0_b_tot, hist_trk_z0_c_tot, hist_trk_z0_btoc_tot], ["bH", "prompt cH", "bH->cH"], cut_string, True, base_filename+"_trk_z0_comp.png")

if __name__ == '__main__':
    main(sys.argv)
