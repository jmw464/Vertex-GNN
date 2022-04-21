#!/usr/bin/env python

################################### visualize_attention.py ####################################
# PURPOSE: 
# EDIT TO: 
# -------------------------------------------Summary-------------------------------------------
# 
###############################################################################################


import dgl
import torch as th
import os,sys,math,glob,ROOT
from ROOT import TH1D, TCanvas, TProfile
import numpy as np
import argparse

import options
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
    args = parser.parse_args()

    runnumber = args.runnumber
    infile_name = args.infile_name
    infile_path = args.data_dir
    outfile_path = args.output_dir

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
    normfile_name = infile_path+infile_name+"_norm"

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
    
    #find how many attention layers the network has
    nattn = 0
    for key in sample_graph.edata.keys():
        if 'attn' in key:
            nattn += 1

    #find how many heads each attention layer has
    nheads = []
    for i in range(nattn):
        nheads.append(sample_graph.edata['attn'+str(i+1)].shape[1])

    profile_attn_true = []
    profile_attn_pred = []
    hist_attn_b = []
    hist_attn_c = []
    hist_attn_none = []
    profile_attn_pt_b = []
    profile_attn_pt_c = []
    profile_attn_pt_none = []
    profile_attn_d0_b = []
    profile_attn_d0_c = []
    profile_attn_d0_none = []
    profile_attn_z0_b = []
    profile_attn_z0_c = []
    profile_attn_z0_none = []

    for i in range(nattn):
        for j in range(nheads[i]):
            profile_attn_true.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_true", ";#alpha_{"+str(i+1)+","+str(j+1)+"};Average edge score", 20, 0, 1))
            profile_attn_pred.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_pred", ";#alpha_{"+str(i+1)+","+str(j+1)+"};Average edge score", 20, 0, 1))
            hist_attn_b.append(TH1D("edg_attn"+str(i+1)+str(j+1)+"_b", ";#alpha_{"+str(i+1)+","+str(j+1)+"};Normalized count", 20, 0, 1))
            hist_attn_c.append(TH1D("edg_attn"+str(i+1)+str(j+1)+"_c", ";#alpha_{"+str(i+1)+","+str(j+1)+"};Normalized count", 20, 0, 1))
            hist_attn_none.append(TH1D("edg_attn"+str(i+1)+str(j+1)+"_none", ";#alpha_{"+str(i+1)+","+str(j+1)+"};Normalized count", 20, 0, 1))
            profile_attn_pt_b.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_pt_b", ";#Sigma pT;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, track_pt_bound[0]*2, track_pt_bound[1]*2))
            profile_attn_pt_c.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_pt_c", ";#Sigma pT;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, track_pt_bound[0]*2, track_pt_bound[1]*2))
            profile_attn_pt_none.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_pt_none", ";#Sigma pT;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, track_pt_bound[0]*2, track_pt_bound[1]*2))
            profile_attn_z0_b.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_z0_b", ";#delta z0;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, 0, track_z0_bound))
            profile_attn_z0_c.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_z0_c", ";#delta z0;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, 0, track_z0_bound))
            profile_attn_z0_none.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_z0_none", ";#delta z0;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, 0, track_z0_bound))
            profile_attn_d0_b.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_d0_b", ";#delta d0;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, 0, track_d0_bound))
            profile_attn_d0_c.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_d0_c", ";#delta d0;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, 0, track_d0_bound))
            profile_attn_d0_none.append(TProfile("edg_attn"+str(i+1)+str(j+1)+"_d0_none", ";#delta d0;Average #alpha_{"+str(i+1)+","+str(j+1)+"};", 50, 0, track_d0_bound))

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
            bin_labels = g.edata['bin_labels'].numpy()
            mult_labels = g.edata['mult_labels'].numpy()
            srcnodes, dstnodes = g.edges()

            for i in range(pred.shape[0]):
                src_pt = abs(1/(features[srcnodes[i],0]*std_features[0]+mean_features[0]))
                dst_pt = abs(1/(features[dstnodes[i],0]*std_features[0]+mean_features[0]))
                src_z0 = features[srcnodes[i],4]*std_features[4]+mean_features[4]
                dst_z0 = features[dstnodes[i],4]*std_features[4]+mean_features[4]
                src_d0 = features[srcnodes[i],4]*std_features[4]+mean_features[4]
                dst_d0 = features[dstnodes[i],4]*std_features[4]+mean_features[4]

                c = 0
                for j in range(nattn):
                    attn = g.edata['attn'+str(j+1)].numpy()
                    for k in range(nheads[j]):
                        profile_attn_true[c].Fill(attn[i,k,0], bin_labels[i])
                        profile_attn_pred[c].Fill(attn[i,k,0], pred[i,0])

                        if mult_labels[i] == 1:
                            hist_attn_b[c].Fill(attn[i,k,0])
                            profile_attn_pt_b[c].Fill(src_pt+dst_pt, attn[i,k,0])
                            profile_attn_z0_b[c].Fill(abs(src_z0-dst_z0), attn[i,k,0])
                            profile_attn_d0_b[c].Fill(abs(src_d0-dst_d0), attn[i,k,0])
                        elif mult_labels[i] == 2:
                            hist_attn_c[c].Fill(attn[i,k,0])
                            profile_attn_pt_c[c].Fill(src_pt+dst_pt, attn[i,k,0])
                            profile_attn_z0_c[c].Fill(abs(src_z0-dst_z0), attn[i,k,0])
                            profile_attn_d0_c[c].Fill(abs(src_d0-dst_d0), attn[i,k,0])
                        else:
                            hist_attn_none[c].Fill(attn[i,k,0])
                            profile_attn_pt_none[c].Fill(src_pt+dst_pt, attn[i,k,0])
                            profile_attn_z0_none[c].Fill(abs(src_z0-dst_z0), attn[i,k,0])
                            profile_attn_d0_none[c].Fill(abs(src_d0-dst_d0), attn[i,k,0])
                        c += 1

    base_filename = outfile_path+runnumber+"/"+infile_name+"_"+runnumber

    canv1 = TCanvas("c1", "c1", 800, 600)

    c = 0
    for i in range(nattn):
        for j in range(nheads[i]):   
            plot_profile(canv1, [profile_attn_true[c], profile_attn_pred[c]], ['true', 'pred'], cut_string, base_filename+"_attn"+str(i)+str(j)+"_tp.png")
            plot_hist(canv1, [hist_attn_b[c], hist_attn_c[c], hist_attn_none[c]], ['b', 'c', 'none'], cut_string, True, False, base_filename+"_attn"+str(i)+str(j)+"_bc.png")
            plot_profile(canv1, [profile_attn_pt_b[c], profile_attn_pt_c[c], profile_attn_pt_none[c]], ['b', 'c', 'none'], cut_string, base_filename+"_attn"+str(i)+str(j)+"_pt.png")
            plot_profile(canv1, [profile_attn_z0_b[c], profile_attn_z0_c[c], profile_attn_z0_none[c]], ['b', 'c', 'none'], cut_string, base_filename+"_attn"+str(i)+str(j)+"_z0.png")
            plot_profile(canv1, [profile_attn_d0_b[c], profile_attn_d0_c[c], profile_attn_d0_none[c]], ['b', 'c', 'none'], cut_string, base_filename+"_attn"+str(i)+str(j)+"_d0.png")
            c += 1


if __name__ == '__main__':
    main(sys.argv)
