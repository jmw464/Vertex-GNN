#!/usr/bin/env python

###################################### process_ntuple.py ######################################
# PURPOSE: convert ntuples from ROOT to HDF5, process truth information to generate track labels
# EDIT TO: add more information from ntuples to output HDF5, change definition of track labels
# -------------------------------------------Summary-------------------------------------------
# This script is run over .root files created by the FlavourTagPerformanceFramework and
# converts them to HDF5 files, extracting all relevant information for GNN training. It
# not only reads in basic tracking parameters, but also matches tracks to their most recent
# HF hadron ancestors in order to determine truth secondary vertices in each jet. Additonally,
# each track is assigned a label based on its respective origin (details can be found in the
# code or other documentation). As the only script in the pre-processing chain, it is kept as
# general as possible (cuts and feature selection is made later on), extracting all
# information that could be relevant to later scripts. This script is intended to run over
# entire ntuples, but there is an optional parameter that can limit the number of events to
# process (useful for testing). There is also a parameter that determines the maximum allowed
# number of events per output file, as these are written in chunks. If one desires to add extra
# GNN features, this needs to be done here (although create_graphs.py would also need to be
# edited at minimum to propagate the changes).
###############################################################################################


import os,sys,math,glob,ROOT
import numpy as np
import h5py
import argparse
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree

from truth_functions import *


def main(argv):
    gROOT.SetBatch(True)
    
    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--ntuple", type=str, required=True, dest="ntuple", help="name of ROOT file to be processed")
    parser.add_argument("-t", "--tree", type=str, default="bTag_AntiKt4EMPFlowJets_BTagging201903", dest="tree", help="name of tree in ntuple")
    parser.add_argument("-i", "--input_dir", type=str, required=True, dest="infile_dir", help="name of input directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="outfile_dir", help="name of output directory")
    parser.add_argument("-e", "--max_events", type=int, default=0, dest="max_events", help="maximum number of events to process")
    parser.add_argument("-v", "--events_per_file", type=int, default=10000, dest="events_per_file", help="number of events per output file")
    args = parser.parse_args()

    ntuple = TFile(args.infile_dir+args.ntuple+".root")
    tree = ntuple.Get(args.tree)
    max_events = args.max_events
    events_per_file = args.events_per_file

    #determine whether vertex weight is contained in ntuple
    sample_entry = tree.GetEntry(0)
    try:
        example = tree.jet_trk_vertex_weight[0][0]
        example = tree.jet_trk_vertex_type[0][0]
        incl_vweight = True
    except:
        print("WARNING: Track to vertex association variables not included in ntuple - files will be generated without this information")
        incl_vweight = False

    #event features
    efeatures = dict()
    efeatures['pv_x'] = []
    efeatures['pv_y'] = []
    efeatures['pv_z'] = []

    #general jet info
    jinfo = dict()
    jinfo['event_no'] = []
    jinfo['jet_no'] = []
    jinfo['ntracks'] = []
    jinfo['jet_flavor'] = []

    #jet features
    jfeatures = dict()
    jfeatures['pt'] = []
    jfeatures['eta'] = []
    jfeatures['phi'] = []

    #general track info
    tinfo = dict()
    tinfo['track_flavor'] = [] #track flavor label
    if incl_vweight: tinfo['vertex_type'] = [] #type of previously associated vertex (related to track to vertex association weight)
    tinfo['hf_ancestor'] = [] #barcode of HF ancestor
    tinfo['hf_pdgid'] = [] #PDG ID of HF ancestor
    tinfo['prev_b_ancestor'] = [] #barcode of previous B ancestor for C tracks (if it exists)
    tinfo['prev_b_pdgid'] = [] #PDG ID of previous B ancestor for C tracks (if it exists)
    tinfo['algo'] = [] #track association with reco algorithms
    tinfo['sv_x'] = []
    tinfo['sv_y'] = []
    tinfo['sv_z'] = []

    #track features
    tfeatures_b = dict()
    tfeatures_b['pt'] = []
    tfeatures_b['eta'] = []
    tfeatures_b['theta'] = []
    tfeatures_b['phi'] = []
    tfeatures_b['d0'] = []
    tfeatures_b['z0'] = []
    tfeatures_b['q'] = []
    tfeatures_e = dict()
    tfeatures_e['cov_d0d0'] = []
    tfeatures_e['cov_z0z0'] = []
    tfeatures_e['cov_phiphi'] = []
    tfeatures_e['cov_thetatheta'] = []
    tfeatures_e['cov_qoverpqoverp'] = []
    tfeatures_c = dict()
    tfeatures_c['cov_d0z0'] = []
    tfeatures_c['cov_d0phi'] = []
    tfeatures_c['cov_d0theta'] = []
    tfeatures_c['cov_d0qoverp'] = []
    tfeatures_c['cov_z0phi'] = []
    tfeatures_c['cov_z0theta'] = []
    tfeatures_c['cov_z0qoverp'] = []
    tfeatures_c['cov_phitheta'] = []
    tfeatures_c['cov_phiqoverp'] = []
    tfeatures_c['cov_thetaqoverp'] = []
    tfeatures_h = dict()
    tfeatures_h['nPixHits'] = []
    tfeatures_h['nSCTHits'] = []
    tfeatures_h['nBLHits'] = []
    tfeatures_h['nPixHoles'] = []
    tfeatures_h['nSCTHoles'] = []
    tfeatures_h['nPixShared'] = []
    tfeatures_h['nSCTShared'] = []
    tfeatures_h['nBLShared'] = []
    tfeatures_h['nPixSplit'] = []
    tfeatures_h['nBLSplit'] = []
    if incl_vweight:
        tfeatures_w = dict()
        tfeatures_w['vweight'] = []

    total_events = tree.GetEntries()
    if max_events == 0 or total_events < max_events:
        max_events = total_events

    #process entries
    ifile = 1
    for ientry,entry in enumerate(tree):
        if ientry < max_events:
            njets = entry.njets
            particle_dict = build_particle_dict(entry)
            primary_vertex = np.array([entry.truth_PVx, entry.truth_PVy, entry.truth_PVz])

            for i in range(njets):
                t_pt = []
                t_eta = []
                t_theta = []
                t_phi = []
                t_d0 = []
                t_z0 = []
                t_q = []
                t_cov_d0d0 = []
                t_cov_z0z0 = []
                t_cov_phiphi = []
                t_cov_thetatheta = []
                t_cov_qoverpqoverp = []
                t_cov_d0z0 = []
                t_cov_d0phi = []
                t_cov_d0theta = []
                t_cov_d0qoverp = []
                t_cov_z0phi = []
                t_cov_z0theta = []
                t_cov_z0qoverp = []
                t_cov_phitheta = []
                t_cov_phiqoverp = []
                t_cov_thetaqoverp = []
                t_nPixHits = []
                t_nSCTHits = []
                t_nBLHits = []
                t_nPixHoles = []
                t_nSCTHoles = []
                t_nPixShared = []
                t_nSCTShared = []
                t_nBLShared = []
                t_nPixSplit = []
                t_nBLSplit = []
                if incl_vweight: t_vweight = []

                t_svx = []
                t_svy = []
                t_svz = []
                t_hf_ancestor = []
                t_hf_pdgid = []
                t_prev_b_ancestor = []
                t_prev_b_pdgid = []
                t_flavor = []
                if incl_vweight: t_vtype = []
                t_algo = []

                ntracks = entry.jet_trk_pt[i].size()
                jet_flavor = entry.jet_LabDr_HadF[i]

                #save relevant feature informationx
                for j in range(ntracks):
                    t_pt.append(entry.jet_trk_pt[i][j]/1000.) #convert from MeV to GeV
                    t_eta.append(entry.jet_trk_eta[i][j])
                    t_theta.append(entry.jet_trk_theta[i][j])
                    t_phi.append(entry.jet_trk_phi[i][j])
                    t_d0.append(entry.jet_trk_d0[i][j])
                    t_z0.append(entry.jet_trk_z0[i][j])
                    t_q.append(entry.jet_trk_charge[i][j])
                    t_cov_d0d0.append(entry.jet_trk_cov_d0d0[i][j])
                    t_cov_z0z0.append(entry.jet_trk_cov_z0z0[i][j])
                    t_cov_phiphi.append(entry.jet_trk_cov_phiphi[i][j])
                    t_cov_thetatheta.append(entry.jet_trk_cov_thetatheta[i][j])
                    t_cov_qoverpqoverp.append(entry.jet_trk_cov_qoverpqoverp[i][j]*1000.*1000.) #convert from keV to MeV
                    t_cov_d0z0.append(entry.jet_trk_cov_d0z0[i][j])
                    t_cov_d0phi.append(entry.jet_trk_cov_d0phi[i][j])
                    t_cov_d0theta.append(entry.jet_trk_cov_d0theta[i][j])
                    t_cov_d0qoverp.append(entry.jet_trk_cov_d0qoverp[i][j]*1000.) #convert from keV to MeV
                    t_cov_z0phi.append(entry.jet_trk_cov_z0phi[i][j])
                    t_cov_z0theta.append(entry.jet_trk_cov_z0theta[i][j])
                    t_cov_z0qoverp.append(entry.jet_trk_cov_z0qoverp[i][j]*1000.) #convert from keV to MeV
                    t_cov_phitheta.append(entry.jet_trk_cov_phitheta[i][j])
                    t_cov_phiqoverp.append(entry.jet_trk_cov_phiqoverp[i][j]*1000.) #convert from keV to MeV
                    t_cov_thetaqoverp.append(entry.jet_trk_cov_thetaqoverp[i][j]*1000.) #convert from keV to MeV
                    t_nPixHits.append(entry.jet_trk_nPixHits[i][j])
                    t_nSCTHits.append(entry.jet_trk_nSCTHits[i][j])
                    t_nBLHits.append(entry.jet_trk_nBLHits[i][j])
                    t_nPixHoles.append(entry.jet_trk_nPixHoles[i][j])
                    t_nSCTHoles.append(entry.jet_trk_nSCTHoles[i][j])
                    t_nPixShared.append(entry.jet_trk_nsharedPixHits[i][j])
                    t_nSCTShared.append(entry.jet_trk_nsharedSCTHits[i][j])
                    t_nBLShared.append(entry.jet_trk_nsharedBLHits[i][j])
                    t_nPixSplit.append(entry.jet_trk_nsplitPixHits[i][j])
                    t_nBLSplit.append(entry.jet_trk_nsplitBLHits[i][j])
                    t_algo.append(entry.jet_trk_algo[i][j])
                    if incl_vweight:
                        t_vtype.append(entry.jet_trk_vertex_type[i][j])
                        t_vweight.append(entry.jet_trk_vertex_weight[i][j]) #0=novtx, 1=pv, 3=pileup

                track_dict = build_track_dict(entry, i, particle_dict, primary_vertex) #initialize track dictionary
                track_dict = group_non_hf_tracks(track_dict) #mark related non HF tracks that share a common vertex - i.e. primary vertex tracks
            
                #save relevant label data
                for ti in track_dict:
                    flavor = track_dict[ti].classification
                    track_sv = np.zeros(3)

                    if flavor == 'b':
                        t_flavor.append(1)
                        track_sv = track_dict[ti].hf_vertex
                    elif flavor == 'c':
                        t_flavor.append(2)
                        track_sv = track_dict[ti].hf_vertex
                    elif flavor == 'btoc':
                        t_flavor.append(3)
                        track_sv = track_dict[ti].prev_b_vertex
                    elif flavor == 'p':
                        t_flavor.append(4)
                    elif flavor == 's':
                        t_flavor.append(5)
                    elif flavor == 'o':
                        t_flavor.append(6)
                    elif flavor == 'nm':
                        t_flavor.append(0)

                    t_hf_ancestor.append(track_dict[ti].hf_ancestor)
                    t_hf_pdgid.append(track_dict[ti].hf_pdgid)
                    t_prev_b_ancestor.append(track_dict[ti].prev_b_ancestor)
                    t_prev_b_pdgid.append(track_dict[ti].prev_b_pdgid)
                    t_svx.append(track_sv[0])
                    t_svy.append(track_sv[0])
                    t_svz.append(track_sv[0])

                #write events
                jinfo['event_no'].append(ientry)
                jinfo['jet_no'].append(i)
                jinfo['ntracks'].append(ntracks)
                jinfo['jet_flavor'].append(jet_flavor)

                jfeatures['pt'].append(entry.jet_pt[i]/1000.) #convert from MeV to GeV
                jfeatures['eta'].append(entry.jet_eta[i])
                jfeatures['phi'].append(entry.jet_phi[i])

                tinfo['hf_ancestor'].extend(t_hf_ancestor)
                tinfo['hf_pdgid'].extend(t_hf_pdgid)
                tinfo['prev_b_ancestor'].extend(t_prev_b_ancestor)
                tinfo['prev_b_pdgid'].extend(t_prev_b_pdgid)
                tinfo['sv_x'].extend(t_svx)
                tinfo['sv_y'].extend(t_svy)
                tinfo['sv_z'].extend(t_svz)
                tinfo['track_flavor'].extend(t_flavor)
                if incl_vweight: tinfo['vertex_type'].extend(t_vtype)
                tinfo['algo'].extend(t_algo)

                tfeatures_b['pt'].extend(t_pt)
                tfeatures_b['eta'].extend(t_eta)
                tfeatures_b['theta'].extend(t_theta)
                tfeatures_b['phi'].extend(t_phi)
                tfeatures_b['d0'].extend(t_d0)
                tfeatures_b['z0'].extend(t_z0)
                tfeatures_b['q'].extend(t_q)
                tfeatures_e['cov_d0d0'].extend(t_cov_d0d0)
                tfeatures_e['cov_z0z0'].extend(t_cov_z0z0)
                tfeatures_e['cov_phiphi'].extend(t_cov_phiphi)
                tfeatures_e['cov_thetatheta'].extend(t_cov_thetatheta)
                tfeatures_e['cov_qoverpqoverp'].extend(t_cov_qoverpqoverp)
                tfeatures_c['cov_d0z0'].extend(t_cov_d0z0)
                tfeatures_c['cov_d0phi'].extend(t_cov_d0phi)
                tfeatures_c['cov_d0theta'].extend(t_cov_d0theta)
                tfeatures_c['cov_d0qoverp'].extend(t_cov_d0qoverp)
                tfeatures_c['cov_z0phi'].extend(t_cov_z0phi)
                tfeatures_c['cov_z0theta'].extend(t_cov_z0theta)
                tfeatures_c['cov_z0qoverp'].extend(t_cov_z0qoverp)
                tfeatures_c['cov_phitheta'].extend(t_cov_phitheta)
                tfeatures_c['cov_phiqoverp'].extend(t_cov_phiqoverp)
                tfeatures_c['cov_thetaqoverp'].extend(t_cov_thetaqoverp)
                tfeatures_h['nPixHits'].extend(t_nPixHits)
                tfeatures_h['nSCTHits'].extend(t_nSCTHits)
                tfeatures_h['nBLHits'].extend(t_nBLHits)
                tfeatures_h['nPixHoles'].extend(t_nPixHoles)
                tfeatures_h['nSCTHoles'].extend(t_nSCTHoles)
                tfeatures_h['nPixShared'].extend(t_nPixShared)
                tfeatures_h['nSCTShared'].extend(t_nSCTShared)
                tfeatures_h['nBLShared'].extend(t_nBLShared)
                tfeatures_h['nPixSplit'].extend(t_nPixSplit)
                tfeatures_h['nBLSplit'].extend(t_nBLSplit)
                if incl_vweight: tfeatures_w['vweight'].extend(t_vweight) 
            
            efeatures['pv_x'].append(entry.truth_PVx)
            efeatures['pv_y'].append(entry.truth_PVy)
            efeatures['pv_z'].append(entry.truth_PVz)

            #output progress
            sys.stdout.write("\rProcessed {} out of {} events".format(ientry+1, max_events))
            sys.stdout.flush()

            #write output file
            if int(ientry+1)%int(events_per_file) == 0 or int(ientry+1) == int(max_events):

                outfile = h5py.File(args.outfile_dir+args.ntuple+"_"+str(ifile).zfill(3)+".hdf5", "w")
                grp_jinfo = outfile.create_group("jinfo")
                grp_jfeatures = outfile.create_group("jfeatures")
                grp_tinfo = outfile.create_group("tinfo")
                grp_tfeatures_b = outfile.create_group("tfeatures_b")
                grp_tfeatures_e = outfile.create_group("tfeatures_e")
                grp_tfeatures_c = outfile.create_group("tfeatures_c")
                grp_tfeatures_h = outfile.create_group("tfeatures_h")
                if incl_vweight: grp_tfeatures_w = outfile.create_group("tfeatures_w")
                grp_efeatures = outfile.create_group("efeatures")

                for k in jinfo.keys():
                    grp_jinfo.create_dataset(k, data = jinfo[k])
                    jinfo[k] = [] #reset dictionary
                for k in jfeatures.keys():
                    jfeatures[k] = np.asarray(jfeatures[k], dtype=np.double)
                    grp_jfeatures.create_dataset(k, data = jfeatures[k])
                    jfeatures[k] = []
                for k in tinfo.keys():
                    grp_tinfo.create_dataset(k, data = tinfo[k])
                    tinfo[k] = []
                for k in tfeatures_b.keys():
                    tfeatures_b[k] = np.asarray(tfeatures_b[k], dtype=np.double)
                    grp_tfeatures_b.create_dataset(k, data = tfeatures_b[k])
                    tfeatures_b[k] = []
                for k in tfeatures_e.keys():
                    tfeatures_e[k] = np.asarray(tfeatures_e[k], dtype=np.double)
                    grp_tfeatures_e.create_dataset(k, data = tfeatures_e[k])
                    tfeatures_e[k] = []
                for k in tfeatures_c.keys():
                    tfeatures_c[k] = np.asarray(tfeatures_c[k], dtype=np.double)
                    grp_tfeatures_c.create_dataset(k, data = tfeatures_c[k])
                    tfeatures_c[k] = []
                for k in tfeatures_h.keys():
                    tfeatures_h[k] = np.asarray(tfeatures_h[k], dtype=np.double)
                    grp_tfeatures_h.create_dataset(k, data = tfeatures_h[k])
                    tfeatures_h[k] = []
                if incl_vweight:
                    for k in tfeatures_w.keys():
                        tfeatures_w[k] = np.asarray(tfeatures_w[k], dtype=np.double)
                        grp_tfeatures_w.create_dataset(k, data = tfeatures_w[k])
                        tfeatures_w[k] = []
                for k in efeatures.keys():
                    efeatures[k] = np.asarray(efeatures[k], dtype=np.double)
                    grp_efeatures.create_dataset(k, data = efeatures[k])
                    efeatures[k] = []

                ifile += 1
                outfile.close()

        else:
            break


if __name__ == '__main__':
    main(sys.argv)
