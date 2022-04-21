#!/usr/bin/env python

###################################### create_graphs.py ######################################
# PURPOSE: generate DGL graph objects from HDF5 files, make cuts on jets and tracks
# EDIT TO: change features used for GNN training, change definitions of edge labels, add
#          additional cuts
# -------------------------------------------Summary-------------------------------------------
# This script is run on HDF5 output from "process_ntuple.py" to produce binary graph files
# compatible with DGL. Cuts and feature selection are also done here, both of which can be
# changed in the "options.py" file. If additional changes to feature selection are desired,
# they can be made by editing this script. While jets that don't meet requirements are cut at
# this stage, all tracks in passing jets are kept (they are just marked for cutting later,
# primarily for the purpose of making plots).
##############################################################################################


import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,ROOT
import numpy as np
import h5py
import argparse
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt
import time

import options
from truth_functions import *


#create the edge list for a complete graph with n nodes
def create_edge_list(n):
    senders = np.zeros(n*(n-1),np.long)
    receivers = np.zeros(n*(n-1),np.long)

    counter = 0
    for i in range(n):
        for j in range(i+1,n):
            senders[counter] = i
            receivers[counter] = j
            senders[counter+1] = j
            receivers[counter+1] = i
            counter += 2

    return senders, receivers


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--ntuple", type=str, required=True, dest="ntuple", help="name of HDF5 file to be processed")
    parser.add_argument("-d", "--dataset", type=str, required=True, dest="dataset", help="name of dataset to be created")
    parser.add_argument("-i", "--input_dir", type=str, required=True, dest="infile_dir", help="name of input directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="outfile_dir", help="name of output directory")
    parser.add_argument("-e", "--max_graphs", type=int, default=0, dest="max_graphs", help="maximum number of graphs to create")
    args = parser.parse_args()

    max_graphs = args.max_graphs

    #input data parameters
    connect_btoc = options.connect_btoc
    incl_errors = options.incl_errors
    incl_corr = options.incl_corr
    incl_hits = options.incl_hits
    incl_vweight = options.incl_vweight 
    jet_pt_cut = options.jet_pt_cut
    jet_eta_cut = options.jet_eta_cut
    track_pt_cut = options.track_pt_cut
    track_eta_cut = options.track_eta_cut
    track_z0_cut = options.track_z0_cut
    vweight_pileup_cut = options.vweight_pileup_cut
    vweight_pv_cut = options.vweight_pv_cut

    nnfeatures_base = 8
    nnfeatures_errors = 5
    nnfeatures_corrs = 10
    nnfeatures_hits = 10

    #file names
    infiles = glob.glob(args.infile_dir+args.ntuple+"_*.hdf5")
    infiles.sort()
    outfiles = []
    for infile in infiles:
        infile_name = os.path.splitext(os.path.basename(infile))[0]
        outfiles.append(args.outfile_dir+"/"+infile_name+".bin")
    
    start_time = time.time()

    print("--------------------------------------------------------------------")

    #check if track to vertex association information is contained in HDF5 file
    infile = h5py.File(infiles[0], "r")
    if 'tfeatures_w' in infile.keys():
        ttv_avail = True
    else:
        ttv_avail = False
        print("WARNING: Track to vertex association information not found")

    total_jets = 0
    for infile_name in infiles:
        infile = h5py.File(infile_name, "r")
        total_jets += len(infile['jinfo']['event_no'])
        infile.close()
    if max_graphs == 0 or max_graphs > total_jets: max_graphs = total_jets
    print("Total number of jets in dataset: {}".format(total_jets))
    print("Maximum number of jets desired: {}".format(max_graphs))

    passed_graphs = cut_graphs = jet_req_cuts = track_req_cuts = tracks_kept = tracks_cut = 0

    #loop through input files
    for ifile, infile_name in enumerate(infiles):

        if passed_graphs >= max_graphs: break #stop reading in new files if maximum desired jet number is reached

        #check if outfile already exists and skip if it's newer than infile unless it's the last prevously processed file (in case more entries are being added)
        if ifile+1 != len(outfiles) and os.path.exists(outfiles[ifile+1]) and os.path.exists(outfiles[ifile]) and os.path.getmtime(outfiles[ifile]) > os.path.getmtime(infile_name):
            ngraphs = len(dgl.load_graphs(outfiles[ifile])[0])
            passed_graphs += ngraphs
            print("Current version of "+os.path.basename(outfiles[ifile])+" already exists and contains "+str(ngraphs)+" jets. Skipping file.")
            continue

        infile = h5py.File(infile_name, "r")
        file_jets = len(infile['jinfo']['event_no'])
        g_list = []

        track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
        event_index = previous_event = -1
        for ientry in range(file_jets):

            #read in event/jet information
            current_event = infile['jinfo']['event_no'][ientry]
            if current_event != previous_event:
                event_index += 1
                previous_event = current_event
                if passed_graphs >= max_graphs: break #stop processing events once specified maximum jet number has been read in

            current_jet = infile['jinfo']['jet_no'][ientry]
            ntracks =  infile['jinfo']['ntracks'][ientry]
            pv_x = infile['efeatures']['pv_x'][event_index]
            pv_y = infile['efeatures']['pv_y'][event_index]
            pv_z = infile['efeatures']['pv_z'][event_index]
            jet_pt = infile['jfeatures']['pt'][ientry]
            jet_eta = infile['jfeatures']['eta'][ientry]
            jet_phi = infile['jfeatures']['phi'][ientry]
            nedges = ntracks*(ntracks-1)

            #apply jet cuts
            if jet_pt > jet_pt_cut and abs(jet_eta) < jet_eta_cut:

                #make jet flavor label definitions consistent
                jet_flavor = infile['jinfo']['jet_flavor'][ientry]
                if jet_flavor == 5: #b-jet
                    jet_flavor = 1
                elif jet_flavor == 4: #c-jet
                    jet_flavor = 2
                elif jet_flavor == 15: #tau-jets
                    jet_flavor = 0
                else:
                    jet_flavor = 0

                node_features_base = np.zeros((ntracks,nnfeatures_base))
                if incl_corr: node_features_corrs = np.zeros((ntracks,nnfeatures_corrs))
                if incl_errors: node_features_errors = np.zeros((ntracks,nnfeatures_errors))
                if incl_hits: node_features_hits = np.zeros((ntracks,nnfeatures_hits))
                if incl_vweight: node_features_vweight = np.zeros((ntracks,1))

                jet_info = np.zeros((ntracks, 4)) #store jet info - jet truth label (0 = l, 1 = b, 2 = c), jet pv coordinates
                track_info = np.zeros((ntracks,4)) #store track general info - track label (see process_ntuples), track sv coordinates
                track_ancestors = np.zeros((ntracks, 4)) #store track ancestor info
                #edge_features = np.zeros((nedges,nefeatures)) 

                #initialize track feature arrays
                hf_ancestors = np.zeros((ntracks,1))
                prev_b_ancestors = np.zeros((ntracks,1))
                track_flavors = np.zeros((ntracks,1))
                reco_use = np.zeros((ntracks,2)) #use of track in SV0, SV1
                passed_cuts = np.zeros((ntracks,1))
                bin_labels = np.zeros((nedges,1))
                mult_labels = np.zeros((nedges,1))
                
                #read in features for each track
                for j in range(ntracks):
                    track_pt  = infile['tfeatures_b']['pt'][track_offset+j]
                    track_eta = infile['tfeatures_b']['eta'][track_offset+j]
                    track_theta = infile['tfeatures_b']['theta'][track_offset+j]
                    track_phi = infile['tfeatures_b']['phi'][track_offset+j]
                    track_d0 = infile['tfeatures_b']['d0'][track_offset+j]
                    track_z0 = infile['tfeatures_b']['z0'][track_offset+j]
                    track_q = infile['tfeatures_b']['q'][track_offset+j]
                    if ttv_avail:
                        track_vweight = infile['tfeatures_w']['vweight'][track_offset+j]
                        track_vtype = infile['tinfo']['vertex_type'][track_offset+j]
                    if incl_errors:
                        track_cov_d0d0 = math.sqrt(infile['tfeatures_e']['cov_d0d0'][track_offset+j])
                        track_cov_z0z0 = math.sqrt(infile['tfeatures_e']['cov_z0z0'][track_offset+j])
                        track_cov_phiphi = math.sqrt(infile['tfeatures_e']['cov_phiphi'][track_offset+j])
                        track_cov_thetatheta = math.sqrt(infile['tfeatures_e']['cov_thetatheta'][track_offset+j])
                        track_cov_qoverpqoverp = math.sqrt(abs(infile['tfeatures_e']['cov_qoverpqoverp'][track_offset+j]))
                    if incl_corr:
                        track_cov_d0z0 = infile['tfeatures_c']['cov_d0z0'][track_offset+j]
                        track_cov_d0phi = infile['tfeatures_c']['cov_d0phi'][track_offset+j]
                        track_cov_d0theta = infile['tfeatures_c']['cov_d0theta'][track_offset+j]
                        track_cov_d0qoverp = infile['tfeatures_c']['cov_d0qoverp'][track_offset+j]
                        track_cov_z0phi = infile['tfeatures_c']['cov_z0phi'][track_offset+j]
                        track_cov_z0theta = infile['tfeatures_c']['cov_z0theta'][track_offset+j]
                        track_cov_z0qoverp = infile['tfeatures_c']['cov_z0qoverp'][track_offset+j]
                        track_cov_phitheta = infile['tfeatures_c']['cov_phitheta'][track_offset+j]
                        track_cov_phiqoverp = infile['tfeatures_c']['cov_phiqoverp'][track_offset+j]
                        track_cov_thetaqoverp = infile['tfeatures_c']['cov_thetaqoverp'][track_offset+j]
                    if incl_hits:
                        track_nPixHits = infile['tfeatures_h']['nPixHits'][track_offset+j]
                        track_nSCTHits = infile['tfeatures_h']['nSCTHits'][track_offset+j]
                        track_nBLHits = infile['tfeatures_h']['nBLHits'][track_offset+j]
                        track_nPixHoles = infile['tfeatures_h']['nPixHoles'][track_offset+j]
                        track_nSCTHoles = infile['tfeatures_h']['nSCTHoles'][track_offset+j]
                        track_nPixShared = infile['tfeatures_h']['nPixShared'][track_offset+j]
                        track_nSCTShared = infile['tfeatures_h']['nSCTShared'][track_offset+j]
                        track_nBLShared = infile['tfeatures_h']['nBLShared'][track_offset+j]
                        track_nPixSplit = infile['tfeatures_h']['nPixSplit'][track_offset+j]
                        track_nBLSplit = infile['tfeatures_h']['nBLSplit'][track_offset+j]

                    track_algo = infile['tinfo']['algo'][track_offset+j]
                    reco_use[j] = [(track_algo & 1 << 2)/4, (track_algo & 1 << 3)/8]

                    hf_ancestors[j] = infile['tinfo']['hf_ancestor'][track_offset+j]
                    hf_pdgid = infile['tinfo']['hf_pdgid'][track_offset+j]
                    prev_b_ancestors[j] = infile['tinfo']['prev_b_ancestor'][track_offset+j]
                    prev_b_pdgid = infile['tinfo']['prev_b_pdgid'][track_offset+j]
                    track_flavors[j] = infile['tinfo']['track_flavor'][track_offset+j]
                    sv_x = infile['tinfo']['sv_x'][track_offset+j]
                    sv_y = infile['tinfo']['sv_y'][track_offset+j]
                    sv_z = infile['tinfo']['sv_z'][track_offset+j]

                    #make cuts on track level
                    if ttv_avail: vertex_condition = (track_vweight < vweight_pv_cut and track_vtype == 1) or (track_vweight < vweight_pileup_cut and track_vtype == 3) or track_vtype == 0
                    else: vertex_condition = True
                    if track_pt > track_pt_cut and abs(track_eta) < track_eta_cut and abs(track_z0) < track_z0_cut and vertex_condition:
                        passed_cuts[j] = 1
                    else:
                        passed_cuts[j] = 0

                    #store information in feature arrays
                    node_features_base[j] = [track_q/track_pt, track_theta, track_phi, track_d0, track_z0, jet_pt, jet_eta, jet_phi]
                    if incl_vweight:
                        node_features_vweight[j] = [track_vweight]
                    if incl_errors:
                        node_features_errors[j] = [track_cov_qoverpqoverp, track_cov_thetatheta, track_cov_phiphi, track_cov_d0d0, track_cov_z0z0]
                    if incl_corr:
                        node_features_corrs[j] = [track_cov_thetaqoverp, track_cov_phiqoverp, track_cov_d0qoverp, track_cov_z0qoverp, track_cov_phitheta, track_cov_d0theta, track_cov_z0theta, track_cov_d0phi, track_cov_z0phi, track_cov_d0z0]
                    if incl_hits:
                        node_features_hits[j] = [track_nPixHits, track_nSCTHits, track_nBLHits, track_nPixHoles, track_nSCTHoles, track_nPixShared, track_nSCTShared, track_nBLShared, track_nPixSplit, track_nBLSplit]
                    track_ancestors[j] = [hf_ancestors[j], hf_pdgid, prev_b_ancestors[j], prev_b_pdgid]
                    track_info[j] = [track_flavors[j], sv_x, sv_y, sv_z]
                    jet_info[j] = [jet_flavor, pv_x, pv_y, pv_z]

                #calculate edge features and truth labels
                counter = 0
                for j in range(ntracks):
                    for k in range(j+1, ntracks):
                        
                        #set edge features
                        #delta_pt = abs(node_features_base[j][0] - node_features_base[k][0])
                        #edge_features[counter:counter+2] = [delta_pt]

                        #truth labels - vertices have to share the same HF ancestor
                        if hf_ancestors[k] == hf_ancestors[j] and hf_ancestors[k] > 0 and track_flavors[j] == 1 and track_flavors[k] == 1: #matching direct ancestors for non secondaries (B to B)
                            bin_labels[counter:counter+2] = 1
                            mult_labels[counter:counter+2] = 1
                        elif hf_ancestors[k] == hf_ancestors[j] and hf_ancestors[k] > 0 and track_flavors[j] == 2 and track_flavors[k] == 2: #matching direct ancestors for non secondaries (prompt C to prompt C)
                            bin_labels[counter:counter+2] = 1
                            mult_labels[counter:counter+2] = 2
                        elif hf_ancestors[k] == hf_ancestors[j] and hf_ancestors[k] > 0 and track_flavors[j] == 3 and track_flavors[k] == 3: #matching direct ancestors for non secondaries (B->C to B->C for same C)
                            bin_labels[counter:counter+2] = 1
                            mult_labels[counter:counter+2] = 1
                        elif prev_b_ancestors[k] == prev_b_ancestors[j] and prev_b_ancestors[k] > 0 and track_flavors[j] == 3 and track_flavors[k] == 3: #matching second ancestors (B->C to B->C for different C)
                            bin_labels[counter:counter+2] = connect_btoc
                            mult_labels[counter:counter+2] = connect_btoc
                        elif ((prev_b_ancestors[k] == hf_ancestors[j] and hf_ancestors[j] > 0) or (prev_b_ancestors[j] == hf_ancestors[k] and hf_ancestors[k] > 0)) and ((track_flavors[j] == 1 and track_flavors[k] == 3) or (track_flavors[j] == 3 and track_flavors[k] == 1)): #matching second ancestor and direct ancestor (B to B->C)
                            bin_labels[counter:counter+2] = connect_btoc
                            mult_labels[counter:counter+2] = connect_btoc
                        counter += 2

                #create graph objects and append them to the list
                if np.sum(passed_cuts) > 1:
                    g = dgl.graph((create_edge_list(ntracks)))
                    g.ndata['features_base'] = th.from_numpy(node_features_base)
                    if incl_vweight: g.ndata['features_vweight'] = th.from_numpy(node_features_vweight)
                    if incl_errors: g.ndata['features_errors'] = th.from_numpy(node_features_errors)
                    if incl_hits: g.ndata['features_hits'] = th.from_numpy(node_features_hits)
                    if incl_corr: g.ndata['features_corr'] = th.from_numpy(node_features_corrs)
                    g.ndata['jet_info'] = th.from_numpy(jet_info)
                    g.ndata['track_info'] = th.from_numpy(track_info)
                    g.ndata['track_ancestors'] = th.from_numpy(track_ancestors)
                    g.ndata['reco_use'] = th.from_numpy(reco_use)
                    g.ndata['passed_cuts'] = th.from_numpy(passed_cuts)
                    g.edata['bin_labels'] = th.from_numpy(bin_labels)
                    g.edata['mult_labels'] = th.from_numpy(mult_labels)
                    g_list.append(g)
                    tracks_kept += np.sum(passed_cuts == 1)
                    tracks_cut += np.sum(passed_cuts == 0)
                    passed_graphs += 1
                else:
                    track_req_cuts += 1
                    cut_graphs += 1

            else:
                jet_req_cuts += 1
                cut_graphs += 1

            track_offset += ntracks

            #output progress
            sys.stdout.write("\rJets processed: {} (Passed: {}, Cut: {}); Files processed: {}/{}".format(cut_graphs+passed_graphs,passed_graphs,cut_graphs,ifile,len(infiles)))
            sys.stdout.flush()

        #save graphs to file
        dgl.save_graphs(outfiles[ifile], g_list)

    print("\nFound enough good jets to reach desired sample size. Finishing up...")
    print("--------------------------------------------------------------------")

    p_time = time.time()-start_time
    print("Graphs cut due to jet requirements: {}".format(jet_req_cuts))
    print("Graphs cut due to track requirements: {}".format(track_req_cuts))
    print("Fraction of tracks cut from passed jets: {}".format(tracks_cut/(tracks_cut+tracks_kept)))
    print("Finished creating graphs. Time elapsed: {}s.".format(p_time))


if __name__ == '__main__':
    main(sys.argv)