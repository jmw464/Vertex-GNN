#!/usr/bin/env python

import os,sys,math,glob,ROOT
import numpy as np
import h5py
import argparse
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree

from truth import *

np.set_printoptions(threshold=sys.maxsize)


#############################################SCRIPT PARAMS#################################################

remove_pv = True
jet_pt_cut = 20000 #20 GeV
jet_eta_cut = 2.5 #edge of detector
track_pt_cut = 650 #650 MeV
track_eta_cut = 2.5 #edge of detector
track_z0_cut = 20

vertex_threshold = 0 #threshold for non HF tracks to be marked as part of the same vertex

###########################################################################################################


def main(argv):
    gROOT.SetBatch(True)
    
    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--ntuple", type=str, required=True, dest="ntuple", help="path of ntuple to be processed")
    parser.add_argument("-t", "--tree", type=str, default="bTag_AntiKt4EMPFlowJets_BTagging201903", dest="tree", help="name of tree in ntuple")
    parser.add_argument("-e", "--entries", type=int, default=1000000, dest="max_entries", help="maximum number of entries to be processed")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="outfile_dir", help="name of directory to save hdf5 file in")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="outfile_name", help="name of dataset to create (without hdf5 extension)")
    args = parser.parse_args()

    max_entries = args.max_entries

    ntuple = TFile(args.ntuple)
    tree = ntuple.Get(args.tree)
   
    outfile = h5py.File(args.outfile_dir+args.outfile_name+".hdf5", "w")
    
    info = dict()
    info['event'] = []
    info['jet'] = []
    info['ntracks'] = []

    jfeatures = dict()
    jfeatures['pt'] = []
    jfeatures['eta'] = []
    jfeatures['phi'] = []

    tfeatures = dict()
    tfeatures['pt'] = []
    tfeatures['eta'] = []
    tfeatures['theta'] = []
    tfeatures['phi'] = []
    tfeatures['d0'] = []
    tfeatures['z0'] = []
    tfeatures['q'] = []

    efeatures = dict()
    efeatures['event_vx'] = []
    efeatures['event_vy'] = []
    efeatures['event_vz'] = []

    labels = dict()
    labels['ancestor'] = []
    labels['flavor'] = []
    labels['second_ancestor'] = [] #only gets determined for B->C tracks

    total_rem_tracks = 0
    total_tracks = 0
    total_jets = 0
    total_rem_jets = 0

    for ientry,entry in enumerate(tree):

        njets = entry.njets
        rem_jets = 0
        
        #check event to see if it can be skipped
        event_pass = False
        for i in range(njets):
            if check_jet(entry, i, jet_pt_cut, jet_eta_cut):
                rem_trk = 0
                nTrack =  entry.jet_trk_pt[i].size()
                for j in range(nTrack):
                    if check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut) and (not remove_pv or not entry.jet_trk_isPV_reco[i][j]):
                        rem_trk += 1
                if rem_trk > 1:
                    event_pass = True
        if not event_pass:
            continue
        
        particle_dict = build_particle_dict(entry)
        primary_vertex = np.array([entry.truth_PVx, entry.truth_PVy, entry.truth_PVz])

        for i in range(njets):
            ntracks = entry.jet_trk_pt[i].size()
            t_pt = []
            t_eta = []
            t_theta = []
            t_phi = []
            t_d0 = []
            t_z0 = []
            t_q = []
            t_ancestor = []
            t_second_ancestor = []
            t_flavor = []

            total_jets += 1

            if check_jet(entry, i, jet_pt_cut, jet_eta_cut):    
                #count tracks that pass cuts to know which jets to skip entirely
                rem_trk = 0
                for j in range(ntracks):
                    if (not remove_pv or not entry.jet_trk_isPV_reco[i][j]) and check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut):
                        rem_trk += 1
                if rem_trk <= 1:
                    continue
                else:
                    total_rem_tracks += rem_trk
                
                #save relevant feature information
                for j in range(ntracks):
                    if not (remove_pv and entry.jet_trk_isPV_reco[i][j]) and check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut):
                        t_pt.append(entry.jet_trk_pt[i][j])
                        t_eta.append(entry.jet_trk_eta[i][j])
                        t_theta.append(entry.jet_trk_theta[i][j])
                        t_phi.append(entry.jet_trk_phi[i][j])
                        t_d0.append(entry.jet_trk_d0[i][j])
                        t_z0.append(entry.jet_trk_z0[i][j])
                        t_q.append(entry.jet_trk_charge[i][j])

                track_dict, jet_cut_trk = build_track_dict(entry, i, particle_dict, remove_pv, track_pt_cut, track_eta_cut, track_z0_cut)
                um_other_tracks = np.array([])

                #make first attempt at track classification (still need to correct some bH labels to bH->cH after this point)
                for ti in track_dict:
                    t_class = classify_track(ti, particle_dict, track_dict)
                    track_dict[ti].classification = t_class
                    if t_class == 'o':
                        um_other_tracks = np.append(um_other_tracks, ti)
                
                #give tracks not associated with HF hadrons unique ancestor barcodes to group them (<0 for vertices not originating from HF hadrons)
                current_ancestor = -1
                for ti in track_dict:
                    for tj in track_dict:
                        vertex_distance = np.linalg.norm(track_dict[ti].vertex - track_dict[tj].vertex)
                        if ti != tj and vertex_distance <= vertex_threshold:
                            if ti in um_other_tracks:
                                track_dict[ti].hf_ancestor = current_ancestor
                                track_dict[tj].hf_ancestor = current_ancestor
                                um_other_tracks = np.delete(um_other_tracks, np.where(um_other_tracks == ti))
                                um_other_tracks = np.delete(um_other_tracks, np.where(um_other_tracks == tj))
                                current_ancestor -= 1
                            elif tj in um_other_tracks:
                                track_dict[tj].hf_ancestor = track_dict[ti].hf_ancestor
                                um_other_tracks = np.delete(um_other_tracks, np.where(um_other_tracks == tj))

                #fix classifications and save relevant label data
                for ti in track_dict:
                    flavor = track_dict[ti].classification
                    if flavor == 'b':
                        t_flavor.append(1)
                    elif flavor == 'c':
                        t_flavor.append(2)
                    elif flavor == 'btoc':
                        t_flavor.append(3)
                    else:
                        t_flavor.append(0)
                    t_ancestor.append(track_dict[ti].hf_ancestor)
                    t_second_ancestor.append(track_dict[ti].btoc_ancestor)

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

                labels['ancestor'].extend(t_ancestor)
                labels['second_ancestor'].extend(t_second_ancestor)
                labels['flavor'].extend(t_flavor)

                info['event'].append(ientry)
                info['jet'].append(i)
                info['ntracks'].append(rem_trk)

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
