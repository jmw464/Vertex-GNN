#!/usr/bin/env python

import os,sys,math,glob,ROOT
import numpy as np
import h5py
import argparse
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree

from truth_functions import *
from plot_functions import plot_hist, plot_bar
import options


def main(argv):
    gROOT.SetBatch(True)
    
    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--ntuple", type=str, required=True, dest="ntuple", help="name of ROOT file to be processed")
    parser.add_argument("-t", "--tree", type=str, default="bTag_AntiKt4EMPFlowJets_BTagging201903", dest="tree", help="name of tree in ntuple")
    parser.add_argument("-e", "--entries", type=int, default=1000000, dest="max_entries", help="maximum number of entries to be processed")
    parser.add_argument("-i", "--input_dir", type=str, required=True, dest="infile_dir", help="name of input directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="outfile_dir", help="name of output directory")
    args = parser.parse_args()

    max_entries = args.max_entries
    ntuple = TFile(args.infile_dir+args.ntuple+".root")
    tree = ntuple.Get(args.tree)
    plot_prefix = args.outfile_dir+args.ntuple
    outfile = h5py.File(args.outfile_dir+args.ntuple+".hdf5", "w")

    #import options from option file
    jet_pt_cut = options.jet_pt_cut
    jet_eta_cut = options.jet_eta_cut
    track_pt_cut = options.track_pt_cut
    track_eta_cut = options.track_eta_cut
    track_z0_cut = options.track_z0_cut
    vweight_pileup_cut = options.vweight_pileup_cut
    vweight_pv_cut = options.vweight_pv_cut
    vertex_threshold = options.vertex_threshold
    incl_errors = options.incl_errors
    incl_corr = options.incl_corr
    incl_hits = options.incl_hits
    incl_vweight = options.incl_vweight

    #general jet info
    info = dict()
    info['event'] = []
    info['jet'] = []
    info['ntracks'] = []
    info['jet_flavor'] = []

    #jet features
    jfeatures = dict()
    jfeatures['pt'] = []
    jfeatures['eta'] = []
    jfeatures['phi'] = []

    #track features
    tfeatures = dict()
    tfeatures['pt'] = []
    tfeatures['eta'] = []
    tfeatures['theta'] = []
    tfeatures['phi'] = []
    tfeatures['d0'] = []
    tfeatures['z0'] = []
    tfeatures['q'] = []
    if incl_vweight:
        tfeatures['vweight'] = []
    if incl_errors:
        tfeatures['cov_d0d0'] = []
        tfeatures['cov_z0z0'] = []
        tfeatures['cov_phiphi'] = []
        tfeatures['cov_thetatheta'] = []
        tfeatures['cov_qoverpqoverp'] = []
    if incl_corr:
        tfeatures['cov_d0z0'] = []
        tfeatures['cov_d0phi'] = []
        tfeatures['cov_d0theta'] = []
        tfeatures['cov_d0qoverp'] = []
        tfeatures['cov_z0phi'] = []
        tfeatures['cov_z0theta'] = []
        tfeatures['cov_z0qoverp'] = []
        tfeatures['cov_phitheta'] = []
        tfeatures['cov_phiqoverp'] = []
        tfeatures['cov_thetaqoverp'] = []
    if incl_hits:
        tfeatures['nPixHits'] = []
        tfeatures['nSCTHits'] = []
        tfeatures['nBLHits'] = []
        tfeatures['nPixHoles'] = []
        tfeatures['nSCTHoles'] = []
        tfeatures['nPixShared'] = []
        tfeatures['nSCTShared'] = []
        tfeatures['nBLShared'] = []
        tfeatures['nPixSplit'] = []
        tfeatures['nBLSplit'] = []

    #event features
    efeatures = dict()
    efeatures['event_vx'] = []
    efeatures['event_vy'] = []
    efeatures['event_vz'] = []

    #track labels and truth info
    labels = dict()
    labels['flavor'] = [] #track flavor label
    labels['ancestor'] = [] #ID of HF ancestor
    labels['ancestor_pdgid'] = [] #PDG ID of HF ancestor
    labels['second_ancestor'] = [] #ID of second HF ancestor (only gets determined for B->C tracks)
    labels['second_ancestor_pdgid'] = [] #PDG ID of second HF ancestor (only gets determined for B->C tracks)
    labels['algo'] = [] #track association with reco algorithms
    labels['track_svx'] = []
    labels['track_svy'] = []
    labels['track_svz'] = []
    labels['passed_cuts'] = []

    total_rem_tracks = total_tracks = total_jets = total_rem_jets = 0

    #process entries
    for ientry,entry in enumerate(tree):
        njets = entry.njets
        rem_jets = 0
        
        #check event to see which jets can be skipped entirely
        passed_jets = []
        for i in range(njets):
            if check_jet(entry, i, jet_pt_cut, jet_eta_cut):
                rem_trk = 0
                nTrack =  entry.jet_trk_pt[i].size()
                for j in range(nTrack):
                    pv_condition = (entry.jet_trk_vertex_weight[i][j] > vweight_pv_cut and entry.jet_trk_vertex_type[i][j] == 1) or (entry.jet_trk_vertex_weight[i][j] > vweight_pileup_cut and entry.jet_trk_vertex_type[i][j] == 2)
                    if check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut) and not pv_condition:
                        rem_trk += 1
                
                if rem_trk > 1:
                    passed_jets.append(i)
        if not len(passed_jets):
            continue
        
        particle_dict = build_particle_dict(entry)
        primary_vertex = np.array([entry.truth_PVx, entry.truth_PVy, entry.truth_PVz])

        for i in range(njets):
            total_jets += 1

            if i in passed_jets:
                
                ntracks = entry.jet_trk_pt[i].size()
                jet_flavor = entry.jet_LabDr_HadF[i]
                rem_trk = 0

                t_pt = []
                t_eta = []
                t_theta = []
                t_phi = []
                t_d0 = []
                t_z0 = []
                t_q = []
                if incl_vweight:
                    t_vweight = []
                if incl_errors:
                    t_cov_d0d0 = []
                    t_cov_z0z0 = []
                    t_cov_phiphi = []
                    t_cov_thetatheta = []
                    t_cov_qoverpqoverp = []
                if incl_corr:
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
                if incl_hits:
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

                t_svx = []
                t_svy = []
                t_svz = []
                t_ancestor = []
                t_ancestor_pdgid = []
                t_second_ancestor = []
                t_second_ancestor_pdgid = []
                t_flavor = []
                t_algo = []
                t_passed_cuts = []
                
                #save relevant feature information
                for j in range(ntracks):

                    t_pt.append(entry.jet_trk_pt[i][j])
                    t_eta.append(entry.jet_trk_eta[i][j])
                    t_theta.append(entry.jet_trk_theta[i][j])
                    t_phi.append(entry.jet_trk_phi[i][j])
                    t_d0.append(entry.jet_trk_d0[i][j])
                    t_z0.append(entry.jet_trk_z0[i][j])
                    t_q.append(entry.jet_trk_charge[i][j])
                    if incl_vweight:
                        t_vweight.append(entry.jet_trk_vertex_weight[i][j])
                    if incl_errors:
                        t_cov_d0d0.append(entry.jet_trk_cov_d0d0[i][j])
                        t_cov_z0z0.append(entry.jet_trk_cov_z0z0[i][j])
                        t_cov_phiphi.append(entry.jet_trk_cov_phiphi[i][j])
                        t_cov_thetatheta.append(entry.jet_trk_cov_thetatheta[i][j])
                        t_cov_qoverpqoverp.append(entry.jet_trk_cov_qoverpqoverp[i][j])
                    if incl_corr:
                        t_cov_d0z0.append(entry.jet_trk_cov_d0z0[i][j])
                        t_cov_d0phi.append(entry.jet_trk_cov_d0phi[i][j])
                        t_cov_d0theta.append(entry.jet_trk_cov_d0theta[i][j])
                        t_cov_d0qoverp.append(entry.jet_trk_cov_d0qoverp[i][j])
                        t_cov_z0phi.append(entry.jet_trk_cov_z0phi[i][j])
                        t_cov_z0theta.append(entry.jet_trk_cov_z0theta[i][j])
                        t_cov_z0qoverp.append(entry.jet_trk_cov_z0qoverp[i][j])
                        t_cov_phitheta.append(entry.jet_trk_cov_phitheta[i][j])
                        t_cov_phiqoverp.append(entry.jet_trk_cov_phiqoverp[i][j])
                        t_cov_thetaqoverp.append(entry.jet_trk_cov_thetaqoverp[i][j])
                    if incl_hits:
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
                    
                    pv_condition = (entry.jet_trk_vertex_weight[i][j] > vweight_pv_cut and entry.jet_trk_vertex_type[i][j] == 1) or (entry.jet_trk_vertex_weight[i][j] > vweight_pileup_cut and entry.jet_trk_vertex_type[i][j] == 2)
                    if not pv_condition and check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut):
                        rem_trk += 1
                        t_passed_cuts.append(1)
                    else:
                        t_passed_cuts.append(0)

                track_dict = build_track_dict(entry, i, particle_dict)
                um_other_tracks = np.array([]) #unmatched "other" tracks

                #perform track classification
                for ti in track_dict:
                    t_class = classify_track(ti, particle_dict, track_dict, primary_vertex)
                    track_dict[ti].classification = t_class
                    if t_class == 'o' or t_class == 'p':
                        um_other_tracks = np.append(um_other_tracks, ti)

                #give tracks not associated with HF hadrons unique ancestor barcodes to group them (<0 for vertices not originating from HF hadrons, -1 is reserved for PV tracks)
                current_ancestor = -2
                for ti in track_dict:
                    for tj in track_dict:
                        vertex_distance = np.linalg.norm(track_dict[ti].vertex - track_dict[tj].vertex)
                        if ti != tj and vertex_distance <= vertex_threshold:
                            if ti in um_other_tracks:
                                if track_dict[ti].classification == 'o': track_dict[ti].hf_ancestor = current_ancestor
                                elif track_dict[ti].classification == 'p': track_dict[ti].hf_ancestor = -1
                                if track_dict[tj].classification == 'o': track_dict[tj].hf_ancestor = current_ancestor
                                elif track_dict[tj].classification == 'p': track_dict[tj].hf_ancestor = -1
                                um_other_tracks = np.delete(um_other_tracks, np.where(um_other_tracks == ti))
                                um_other_tracks = np.delete(um_other_tracks, np.where(um_other_tracks == tj))
                                current_ancestor -= 1
                            elif tj in um_other_tracks:
                                if track_dict[tj].classification == 'o': track_dict[tj].hf_ancestor = track_dict[ti].hf_ancestor
                                elif track_dict[tj].classification == 'p': track_dict[tj].hf_ancestor = -1
                                um_other_tracks = np.delete(um_other_tracks, np.where(um_other_tracks == tj))
            
                #save relevant label data
                for ti in track_dict:
                    flavor = track_dict[ti].classification
                    ancestor_pdgid = 0
                    second_ancestor_pdgid = 0

                    if flavor == 'b':
                        t_flavor.append(1)
                        ancestor_pdgid = particle_dict[track_dict[ti].hf_ancestor].pdgid 
                    elif flavor == 'c':
                        t_flavor.append(2)
                        ancestor_pdgid = particle_dict[track_dict[ti].hf_ancestor].pdgid 
                    elif flavor == 'btoc':
                        t_flavor.append(3)
                        ancestor_pdgid = particle_dict[track_dict[ti].hf_ancestor].pdgid 
                        second_ancestor_pdgid = particle_dict[track_dict[ti].btoc_ancestor].pdgid
                    elif flavor == 'p':
                        t_flavor.append(4)
                    elif flavor == 's':
                        t_flavor.append(5)
                    elif flavor == 'o':
                        t_flavor.append(6)
                    elif flavor == 'nm':
                        t_flavor.append(0)

                    t_ancestor.append(track_dict[ti].hf_ancestor)
                    t_ancestor_pdgid.append(ancestor_pdgid)
                    t_second_ancestor.append(track_dict[ti].btoc_ancestor)
                    t_second_ancestor_pdgid.append(second_ancestor_pdgid)
                    t_svx.append(track_dict[ti].ancestor_vertex[0])
                    t_svy.append(track_dict[ti].ancestor_vertex[1])
                    t_svz.append(track_dict[ti].ancestor_vertex[2])

                #write events
                jfeatures['pt'].append(entry.jet_pt[i])
                jfeatures['eta'].append(entry.jet_eta[i])
                jfeatures['phi'].append(entry.jet_phi[i])

                tfeatures['pt'].extend(t_pt)
                tfeatures['eta'].extend(t_eta)
                tfeatures['theta'].extend(t_theta)
                tfeatures['phi'].extend(t_phi)
                tfeatures['d0'].extend(t_d0)
                tfeatures['z0'].extend(t_z0)
                tfeatures['q'].extend(t_q)
                if incl_vweight:
                    tfeatures['vweight'].extend(t_vweight)
                if incl_errors:
                    tfeatures['cov_d0d0'].extend(t_cov_d0d0)
                    tfeatures['cov_z0z0'].extend(t_cov_z0z0)
                    tfeatures['cov_phiphi'].extend(t_cov_phiphi)
                    tfeatures['cov_thetatheta'].extend(t_cov_thetatheta)
                    tfeatures['cov_qoverpqoverp'].extend(t_cov_qoverpqoverp)
                if incl_corr:
                    tfeatures['cov_d0z0'].extend(t_cov_d0z0)
                    tfeatures['cov_d0phi'].extend(t_cov_d0phi)
                    tfeatures['cov_d0theta'].extend(t_cov_d0theta)
                    tfeatures['cov_d0qoverp'].extend(t_cov_d0qoverp)
                    tfeatures['cov_z0phi'].extend(t_cov_z0phi)
                    tfeatures['cov_z0theta'].extend(t_cov_z0theta)
                    tfeatures['cov_z0qoverp'].extend(t_cov_z0qoverp)
                    tfeatures['cov_phitheta'].extend(t_cov_phitheta)
                    tfeatures['cov_phiqoverp'].extend(t_cov_phiqoverp)
                    tfeatures['cov_thetaqoverp'].extend(t_cov_thetaqoverp)
                if incl_hits:
                    tfeatures['nPixHits'].extend(t_nPixHits)
                    tfeatures['nSCTHits'].extend(t_nSCTHits)
                    tfeatures['nBLHits'].extend(t_nBLHits)
                    tfeatures['nPixHoles'].extend(t_nPixHoles)
                    tfeatures['nSCTHoles'].extend(t_nSCTHoles)
                    tfeatures['nPixShared'].extend(t_nPixShared)
                    tfeatures['nSCTShared'].extend(t_nSCTShared)
                    tfeatures['nBLShared'].extend(t_nBLShared)
                    tfeatures['nPixSplit'].extend(t_nPixSplit)
                    tfeatures['nBLSplit'].extend(t_nBLSplit)
                
                labels['ancestor'].extend(t_ancestor)
                labels['ancestor_pdgid'].extend(t_ancestor_pdgid)
                labels['second_ancestor'].extend(t_second_ancestor)
                labels['second_ancestor_pdgid'].extend(t_second_ancestor_pdgid)
                labels['track_svx'].extend(t_svx)
                labels['track_svy'].extend(t_svy)
                labels['track_svz'].extend(t_svz)
                labels['flavor'].extend(t_flavor)
                labels['algo'].extend(t_algo)
                labels['passed_cuts'].extend(t_passed_cuts)

                info['event'].append(ientry)
                info['jet'].append(i)
                info['ntracks'].append(ntracks)
                info['jet_flavor'].append(jet_flavor)

                total_rem_tracks += rem_trk
                total_tracks += ntracks
                rem_jets += 1
           
        efeatures['event_vx'].append(entry.truth_PVx)
        efeatures['event_vy'].append(entry.truth_PVy)
        efeatures['event_vz'].append(entry.truth_PVz)

        total_rem_jets += rem_jets

        #output progress
        sys.stdout.write("\rProcessed {} jets".format(total_rem_jets))
        sys.stdout.flush()

        if total_rem_jets >= max_entries:
            break

    sys.stdout.write("\rFinished processing. Total jets used from sample: {}".format(total_rem_jets))
    sys.stdout.flush()
    print("\nCut {} jets and {}% of {} tracks in remaining jets".format(total_jets-total_rem_jets, (total_tracks-total_rem_tracks)*100./total_tracks, total_tracks))

    grp_info = outfile.create_group("info")
    grp_tfeatures = outfile.create_group("tfeatures")
    grp_jfeatures = outfile.create_group("jfeatures")
    grp_efeatures = outfile.create_group("efeatures")
    grp_labels = outfile.create_group("labels")

    for k in info.keys():
        grp_info.create_dataset(k, data = info[k])
    for k in tfeatures.keys():
        tfeatures[k] = np.asarray(tfeatures[k], dtype=np.double)
        grp_tfeatures.create_dataset(k, data = tfeatures[k])
    for k in jfeatures.keys():
        jfeatures[k] = np.asarray(jfeatures[k], dtype=np.double)
        grp_jfeatures.create_dataset(k, data = jfeatures[k])
    for k in efeatures.keys():
        efeatures[k] = np.asarray(efeatures[k], dtype=np.double)
        grp_efeatures.create_dataset(k, data = efeatures[k])
    for k in labels.keys():
        grp_labels.create_dataset(k, data = labels[k])

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
