#!/usr/bin/env python

##################################### truth_functions.py #####################################
# PURPOSE: contains helper functions related to MC truth (used in processing)
# EDIT TO: /
# ------------------------------------------Summary-------------------------------------------
# This script contains a collection of functions used in the earlier steps of data processing
# (specifically in process_ntuple.py and create_graphs.py). It also contains class definitions
# for the truth_particle and truth_track objects used throughout processing.
##############################################################################################


import numpy as np
import ROOT


wd_cm = [411, 421, 431] #weakly decaying charm meson PDGIDs
wd_cb = [4122, 4132, 4212, 4232, 4332] #weakly decaying charm baryon PDGIDs
wd_bm = [511, 521, 531, 541] #weakly decaying bottom meson PDGIDs
wd_bb = [5112, 5122, 5132, 5212, 5222, 5232, 5332] #weakly decaying bottom baryon PDGIDs


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

    def print_particle(self,level):
        prefix = "--"*level + ">"
        print("{} PDG ID: {}, Barcode: {}, Parents: {}, Children: {}, PV: {}, DV: {}".format(prefix, self.pdgid, self.barcode, self.parents, self.children, self.pv, self.dv))


class truth_track():
    def __init__(self, barcode, pdgid, vertex, pt, eta, phi, d0, z0):
        self.barcode = barcode
        self.pdgid = pdgid
        self.vertex = vertex
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.d0 = d0
        self.z0 = z0

        #initialize track origin information
        self.hf_ancestor = 0 #direct HF ancestor barcode
        self.hf_pdgid = 0 #direct HF ancestor PDGID
        self.hf_vertex = np.zeros(3)
        self.prev_b_ancestor = 0 #previous B ancestor if direct ancestor is not B
        self.prev_b_pdgid = 0 #previous B ancestor PDGID if direct ancestor is not B
        self.prev_b_vertex = np.zeros(3)
        self.classification = "" #track classification based on origin

    def print_track(self,level):
        prefix = "xx"*level + ">"
        if self.classification: print("{} PDG ID: {}, Barcode: {}, Vertex: {}, d0: {}, z0: {}, Class: {}".format(prefix, self.pdgid, self.barcode, self.vertex, self.d0, self.z0, self.classification))
        else: print("{} PDG ID: {}, Barcode: {}, Vertex: {}, d0: {}, z0: {}".format(prefix, self.pdgid, self.barcode, self.vertex, self.d0, self.z0)) 


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


def build_track_dict(entry, i, particle_dict, primary_vertex):
    track_dict = {}
    nTrack =  entry.jet_trk_pt[i].size()

    #set up basic track dictionary
    for j in range(nTrack):
        trk_vertex = np.array([entry.jet_trk_vtx_X[i][j], entry.jet_trk_vtx_Y[i][j], entry.jet_trk_vtx_Z[i][j]])
        trk_pt = entry.jet_trk_pt[i][j]
        trk_eta = entry.jet_trk_eta[i][j]
        trk_phi = entry.jet_trk_phi[i][j]
        trk_pdgId = entry.jet_trk_pdg_id[i][j]
        trk_barcode = entry.jet_trk_barcode[i][j]
        trk_d0 = entry.jet_trk_d0[i][j]
        trk_z0 = entry.jet_trk_z0[i][j]

        track_dict[j] = truth_track(trk_barcode, trk_pdgId, trk_vertex, trk_pt, trk_eta, trk_phi, trk_d0, trk_z0)

    #evaluate ancestors of tracks in dictionary
    for ti in track_dict:
        track = track_dict[ti]

        #can only determine ancestors of tracks with matched truth particles
        if track.pdgid > 0:
            t_barcode = track.barcode
            track_particle = particle_dict[t_barcode]

            #get direct HF ancestors of track particle
            ancestors = get_hf_relatives(track_particle, particle_dict, np.array([]), 'a', '', 0)
            ancestors = np.unique(ancestors) #only keep unique barcodes

            #check which HF ancestor will be saved in case there are multiple direct HF ancestors (by checking distance between HF DV and track PV)
            min_distance = -1
            direct_ancestor = 0
            for ancestor in ancestors:
                distance = np.linalg.norm(track_particle.pv - particle_dict[ancestor].dv)
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance
                    track.hf_ancestor = ancestor
                    track.hf_pdgid = particle_dict[ancestor].pdgid
                    track.hf_vertex = particle_dict[ancestor].dv

            #get previous B ancestors of track particle if there is a direct C ancestor
            if id_particle(track.hf_pdgid) == 'c':
                ancestor_particle = particle_dict[track.hf_ancestor]
                prev_b_ancestors = get_hf_relatives(ancestor_particle, particle_dict, np.array([]), 'a', 'b', 1)

                #check for closest B ancestor if there are multiple
                min_distance = -1
                prev_b_ancestor = 0
                for prev_b_ancestor in prev_b_ancestors:
                    distance = np.linalg.norm(ancestor_particle.pv - particle_dict[prev_b_ancestor].dv)
                    if (min_distance == -1 or distance < min_distance):
                        min_distance = distance
                        track.prev_b_ancestor = prev_b_ancestor
                        track.prev_b_pdgid = particle_dict[prev_b_ancestor].pdgid
                        track.prev_b_vertex = particle_dict[prev_b_ancestor].dv

        track.classification = classify_track(track, primary_vertex)
        track_dict[ti] = track

    return track_dict


def id_particle(pdgid):
    if abs(pdgid) in wd_cm or abs(pdgid) in wd_cb:
        return 'c'
    elif abs(pdgid) in wd_bm or abs(pdgid) in wd_bb:
        return 'b'
    else:
        return 'o'


#function to retrieve most recent independent (non-interrelated) HF relatives (descendants or antecedents) of a given particle
def get_hf_relatives(particle, particle_dict, particle_list, mode, flavor, first): #first is used to mark the initial call of the function so it doesn't terminate immediately if called on HF particle
    barcode = particle.barcode
    pdgid = particle.pdgid
    label = id_particle(pdgid)
    
    if not first and ((flavor != 'b' and label == 'c') or (flavor != 'c' and label == 'b')):
        particle_list = np.append(particle_list, barcode)
        return particle_list #stop searching branch if HF relative is found
    
    if mode == 'a':
        for parent in particle.parents:
            particle_list = get_hf_relatives(particle_dict[parent], particle_dict, particle_list, mode, flavor, 0)
    elif mode == 'd':
        for child in particle.children:
            particle_list = get_hf_relatives(particle_dict[child], particle_dict, particle_list, mode, flavor, 0)

    return particle_list


def classify_track(track, primary_vertex):
    barcode = track.barcode
    hf_barcode = track.hf_ancestor
    hf_pdgid = track.hf_pdgid
    prev_b_barcode = track.prev_b_ancestor
    prev_b_pdgid = track.prev_b_pdgid

    track_vertex = track.vertex #track production vertex
    if hf_barcode > 0:
        sv_distance = np.linalg.norm(track_vertex-track.hf_vertex)

    if barcode < -990:
        return 'nm' #track not matched to truth particle
    elif barcode >= 200000:
        return 's' #track originating from secondary interaction (in detector simulation rather than original event simulation)
    elif hf_barcode > 0 and id_particle(hf_pdgid) == 'c':
        if prev_b_barcode > 0:
            return 'btoc' #track originating from c hadron via b hadron
        else:
            return 'c' #track originating from c hadron without b ancestors
    elif hf_barcode > 0 and id_particle(hf_pdgid) == 'b':
        return 'b' #track originating from b hadron
    elif np.linalg.norm(track_vertex-primary_vertex) < 1e-3:
        return 'p' #track originating from primary vertex
    else:
        return 'o' #track originating from other particle


#function used to mark non HF tracks as being related based on certain criteria - they will have negative HF ancestor barcodes
def group_non_hf_tracks(track_dict):
    non_hf_tracks = np.array([]) #array containing tracks not originating from HF particles
    for ti in track_dict:
        t_class = track_dict[ti].classification
        if t_class == 'o' or t_class == 'p' or (t_class == 's' and track_dict[ti].hf_ancestor == 0): #check p, o and s tracks (s only if not originating from HF hadron)
            non_hf_tracks = np.append(non_hf_tracks, ti)

    #give tracks not associated with HF hadrons unique ancestor barcodes to group them (<0 for vertices not originating from HF hadrons, -1 is reserved for PV tracks)
    current_ancestor = -2
    for ti in track_dict:
        for tj in track_dict:
            vertex_distance = np.linalg.norm(track_dict[ti].vertex - track_dict[tj].vertex)
            if ti != tj and vertex_distance < 1e-3: #mark tracks as connected that have the same origin vertex
                if ti in non_hf_tracks:
                    if track_dict[ti].classification == 'o' or track_dict[ti].classification == 's': track_dict[ti].hf_ancestor = current_ancestor
                    elif track_dict[ti].classification == 'p': track_dict[ti].hf_ancestor = -1
                    if track_dict[tj].classification == 'o' or track_dict[tj].classification == 's': track_dict[tj].hf_ancestor = current_ancestor
                    elif track_dict[tj].classification == 'p': track_dict[tj].hf_ancestor = -1
                    non_hf_tracks = np.delete(non_hf_tracks, np.where(non_hf_tracks == ti))
                    non_hf_tracks = np.delete(non_hf_tracks, np.where(non_hf_tracks == tj))
                    current_ancestor -= 1
                elif tj in non_hf_tracks:
                    if track_dict[tj].classification == 'o' or track_dict[tj].classification == 's': track_dict[tj].hf_ancestor = track_dict[ti].hf_ancestor
                    elif track_dict[tj].classification == 'p': track_dict[tj].hf_ancestor = -1
                    non_hf_tracks = np.delete(non_hf_tracks, np.where(non_hf_tracks == tj))

    return track_dict


#prints particle/track dictionary in tree form starting from specified particle - useful for examining individual events
def print_tree(particle, particle_dict, track_dict, level): #level = 0 disables recursive printing (only prints given particle and track info)
    barcode = particle.barcode
    pdgid = particle.pdgid
    condition = True #eg pdgid != 21 and pdgid != 22 OR (id_particle(pdgid) == 'ch') or (id_particle(pdgid) == 'bh') if only certain types of particles should be printed

    #print information
    if condition:
        for ti in track_dict:
            if barcode == track_dict[ti].barcode:
                track_dict[ti].print_track(level)
        particle.print_particle(level)

    #loop through children recursively
    if level:
        for child in particle.children:
            print_tree(particle_dict[child], particle_dict, track_dict, level+1)


#implement cuts on jet level - NOT CURRENTLY USED
def check_jet(entry, jet, pt_cut, eta_cut):
    if entry.jet_pt[jet] > pt_cut and abs(entry.jet_eta[jet]) < eta_cut:
        return True
    else:
        return False


#implement cuts on track level - NOT CURRENTLY USED
def check_track(entry, jet, track, pt_cut, eta_cut, z0_cut):
    if entry.jet_trk_pt[jet][track] > pt_cut and abs(entry.jet_trk_eta[jet][track]) < eta_cut and abs(entry.jet_trk_z0[jet][track]) < z0_cut:
        return True
    else:
        return False