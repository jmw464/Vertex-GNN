#!/usr/bin/env python

import os,sys,math,ROOT,glob
import numpy as np
import argparse
from ROOT import TFile, TH1D, TH1I, gROOT, TCanvas, gPad, TLegend
import time


class truth_particle():
    def __init__(self, barcode, pdgid, pv, dv, charged, p, parents, children):
        self.barcode = barcode
        self.pdgid = pdgid
        self.pv = pv
        self.dv = dv
        self.charged = charged
        self.p = p
        self.parents = parents
        self.children = children


class truth_track():
    def __init__(self, barcode, pdgid, vertex, pt, eta, phi, z0):
        self.barcode = barcode
        self.pdgid = pdgid
        self.vertex = vertex
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.z0 = z0
        self.hf_ancestor = 0 #overall HF ancestor (always corresponds to b hadron for bH->cH tracks)
        self.btoc_ancestor = 0 #direct charm ancestor for bH->cH tracks from charm hadrons
        self.classification = ""


def build_particle_dict(entry):
    particle_dict = {}
    nTruth = entry.truth_pdgId.size()
    for i in range ( nTruth ):
        truth_pdgId = entry.truth_pdgId[i]
        truth_barcode = entry.truth_barcode[i]
        truth_status = entry.truth_status[i]
        truth_charged = entry.truth_isCharged[i]

        truth_pv = np.array([entry.truth_pvtx_x[i], entry.truth_pvtx_y[i], entry.truth_pvtx_z[i]])
        truth_dv = np.array([entry.truth_dvtx_x[i], entry.truth_dvtx_y[i], entry.truth_dvtx_z[i]])
        truth_p = np.array([entry.truth_px[i], entry.truth_py[i], entry.truth_pz[i]])

        truth_nParent = entry.truth_parent_pdgId[i].size()
        truth_nChild = entry.truth_child_pdgId[i].size()

        child_pdgId_array = np.zeros(truth_nChild)
        child_barcode_array = np.zeros(truth_nChild)
        for j in range (truth_nChild):
            child_pdgId = entry.truth_child_pdgId[i][j]
            child_pdgId_array[j] = child_pdgId
            child_barcode = entry.truth_child_barcode[i][j]
            child_barcode_array[j] = child_barcode

        parent_pdgId_array = np.zeros(truth_nParent)
        parent_barcode_array = np.zeros(truth_nParent)
        for j in range (truth_nParent):
            parent_pdgId = entry.truth_parent_pdgId[i][j]
            parent_pdgId_array[j] = parent_pdgId                
            parent_barcode = entry.truth_parent_barcode[i][j]
            parent_barcode_array[j] = parent_barcode

        particle_dict[truth_barcode] = truth_particle(truth_barcode, truth_pdgId, truth_pv, truth_dv, truth_charged, truth_p, parent_barcode_array, child_barcode_array)

    return particle_dict


def build_track_dict(entry, i, particle_dict, remove_pv, track_pt_cut, track_eta_cut, track_z0_cut):
    track_dict = {}
    nTrack =  entry.jet_trk_pt[i].size()
    jet_cut_trk = 0

    for j in range(nTrack):
        trk_vertex = np.array([entry.jet_trk_vtx_X[i][j], entry.jet_trk_vtx_Y[i][j], entry.jet_trk_vtx_Z[i][j]])

        if check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut) and (not remove_pv or not entry.jet_trk_isPV_reco[i][j]):
            trk_pt = entry.jet_trk_pt[i][j]
            trk_eta = entry.jet_trk_eta[i][j]
            trk_phi = entry.jet_trk_phi[i][j]
            trk_pdgId = entry.jet_trk_pdg_id[i][j]
            trk_barcode = entry.jet_trk_barcode[i][j]
            trk_z0 = entry.jet_trk_z0[i][j]

            track_dict[j] = truth_track(trk_barcode, trk_pdgId, trk_vertex, trk_pt, trk_eta, trk_phi, trk_z0)

        else:
            jet_cut_trk += 1

    for ti in track_dict:
        track = track_dict[ti]

        #don't process tracks that don't have associated truth particles
        if track.pdgid == -999:
            continue

        #get direct HF ancestors of track particle
        t_barcode = track.barcode
        track_particle = particle_dict[t_barcode]
        ancestors = np.array([])
        ancestors = get_hf_relatives(track_particle, particle_dict, ancestors, 'a', 0)
        ancestors = np.unique(ancestors) #only keep unique barcodes

        #check which ancestor will be marked as primary (by checking distance between HF DV and track PV)
        min_distance = -1
        direct_ancestor = 0
        for ancestor in ancestors:
            distance = np.linalg.norm(track_particle.pv - particle_dict[ancestor].dv)
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                direct_ancestor = ancestor
        track.hf_ancestor = direct_ancestor
        track_dict[ti] = track

    return track_dict, jet_cut_trk


def id_particle(pdgid):
    wd_cm = [411, 421, 431] #weakly decaying charm mesons
    wd_cb = [4122, 4132, 4232, 4212, 4332] #weakly decaying charm baryons
    wd_bm = [511, 521, 531, 541] #weakly decaying bottom mesons
    wd_bb = [5122, 5132, 5232, 5112, 5212, 5222, 5332] #weakly decaying bottom baryons

    if abs(pdgid) in wd_cm or abs(pdgid) in wd_cb:
        return 'ch'
    elif abs(pdgid) in wd_bm or abs(pdgid) in wd_bb:
        return 'bh'
    else:
        return 'other'


#function to retrieve most recent independent (non-interrelated) HF relatives (descendants or antecedents) of a given particle
def get_hf_relatives(particle, particle_dict, particle_list, mode, skip): #skip is used when examining the HF ancestors of a HF hadron (so the function can enter the recursive part)
    barcode = particle.barcode
    pdgid = particle.pdgid
    label = id_particle(pdgid)
    
    if not skip and (label == 'ch' or label == 'bh'):
        particle_list = np.append(particle_list, barcode)
        return particle_list #stop searching branch if HF relative is found
    
    if mode == 'a':
        for parent in particle.parents:
            particle_list = get_hf_relatives(particle_dict[parent], particle_dict, particle_list, mode, 0)
    elif mode == 'd':
        for child in particle.children:
            particle_list = get_hf_relatives(particle_dict[child], particle_dict, particle_list, mode, 0)

    return particle_list


def classify_track(index, particle_dict, track_dict, threshold_dist, bc_threshold_dist):
    barcode = track_dict[index].barcode
    hf_barcode = track_dict[index].hf_ancestor
    track_vertex = track_dict[index].vertex
    if hf_barcode > 0: primary_distance = np.linalg.norm(track_vertex-particle_dict[hf_barcode].dv)

    min_distance = -1
    second_ancestor = 0
    if barcode < -990:
        return 'nm' #no truth particle match
    elif hf_barcode > 0 and id_particle(particle_dict[hf_barcode].pdgid) == 'ch' and primary_distance <= threshold_dist:
        ch_barcodes = get_hf_relatives(particle_dict[hf_barcode], particle_dict, np.array([]), 'a', 1)
        for ch_barcode in ch_barcodes:
            distance = np.linalg.norm(particle_dict[hf_barcode].pv - particle_dict[ch_barcode].dv)
            if (min_distance == -1 or distance < min_distance) and id_particle(particle_dict[ch_barcode].pdgid) == 'bh':
                min_distance = distance
                second_ancestor = ch_barcode
        if second_ancestor > 0 and min_distance <= bc_threshold_dist:
            track_dict[index].btoc_ancestor = track_dict[index].hf_ancestor
            track_dict[index].hf_ancestor = second_ancestor
            return 'btoc'
        else:
            return 'c'
    elif hf_barcode > 0 and id_particle(particle_dict[hf_barcode].pdgid) == 'bh' and primary_distance <= threshold_dist:
        #bh_barcodes = get_hf_relatives(particle_dict[hf_barcode], particle_dict, np.array([]), 'd', 1)
        #for bh_barcode in bh_barcodes:
        #    distance = np.linalg.norm(particle_dict[hf_barcode].dv - particle_dict[bh_barcode].pv)
        #    if (min_distance == -1 or distance < min_distance) and id_particle(particle_dict[bh_barcode].pdgid) == 'ch':
        #        min_distance = distance
        #        second_ancestor = bh_barcode
        #if second_ancestor != -1 and min_distance <= bc_threshold_dist:
        #    return 'btoc'
        #else:
        return 'b'
    else:
        return 'o'


#implement cuts on jet level
def check_jet(entry, jet, pt_cut, eta_cut):
    if entry.jet_pt[jet] > pt_cut and abs(entry.jet_eta[jet]) < eta_cut:
        return True
    else:
        return False


#implement cuts on track level
def check_track(entry, jet, track, pt_cut, eta_cut, z0_cut):
    if entry.jet_trk_pt[jet][track] > pt_cut and abs(entry.jet_trk_eta[jet][track]) < eta_cut and abs(entry.jet_trk_z0[jet][track]) < z0_cut:
        return True
    else:
        return False
