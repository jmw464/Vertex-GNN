#!/usr/bin/env python

import os,sys,math,ROOT,glob
import numpy as np
from ROOT import TFile, TH1D, gROOT, TCanvas, gPad, TLegend

maxentries = 5000
remove_pv = True

#set thresholds for track classification
threshold_dist = 50 #maximum distance between HF vertex and track vertex
threshold_level = 5 #maximum degrees of removal between HF particle and track particle (or bH and cH for bH->cH)

ntuple = TFile("/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/raw/user.jmwagner.24900045.Akt4EMPf_BTagging201903._000007.root")
savepath = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/"
dataname = "btag07_19"

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

    def get_pdgid(self):
        return self.pdgid

    def get_barcode(self):
        return self.barcode
    
    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def get_dv(self):
        return self.dv

    def get_pv(self):
        return self.pv

    def get_p(self):
        return self.p

    def is_charged(self):
        return self.charged


class truth_track():
    def __init__(self, barcode, pdgid, vertex, pt, eta, phi):
        self.barcode = barcode
        self.pdgid = pdgid
        self.vertex = vertex
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.bh_ancestors = np.array([])
        self.ch_ancestors = np.array([])
        self.classification = ""

    def get_pdgid(self):
        return self.pdgid

    def get_barcode(self):
        return self.barcode

    def get_vertex(self):
        return self.vertex

    def get_pt(self):
        return self.pt

    def get_eta(self):
        return self.eta

    def get_phi(self):
        return self.phi
    
    def add_bh_ancestor(self, barcode):
        self.bh_ancestors = np.append(self.bh_ancestors, barcode)
        self.bh_ancestors = np.reshape(self.bh_ancestors, (int(len(self.bh_ancestors)/2),2))

    def add_ch_ancestor(self, barcode):
        self.ch_ancestors = np.append(self.ch_ancestors, barcode)
        self.ch_ancestors = np.reshape(self.ch_ancestors, (int(len(self.ch_ancestors)/2),2))

    def set_classification(self, classification):
        self.classification = classification

    def get_bh_ancestors(self):
        return self.bh_ancestors

    def get_ch_ancestors(self):
        return self.ch_ancestors

    def get_classification(self):
        return self.classification


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


def get_ancestors(particle, particle_dict, particle_list, level): #particle_list stores barcodes to avoid double counting
    barcode = particle.get_barcode()
    pdgid = particle.get_pdgid()

    #if barcode not in particle_list:
    particle_list = np.append(particle_list, np.array([barcode, level]))
        
    for parent in particle.get_parents():
        particle_list = get_ancestors(particle_dict[parent], particle_dict, particle_list, level+1)
    
    return particle_list


def classify_track(barcode, particle_dict, track_dict):
    bh_ancestors = track_dict[barcode].get_bh_ancestors()
    ch_ancestors = track_dict[barcode].get_ch_ancestors()
    track_vertex = track_dict[barcode].get_vertex()

    if track_vertex[0] < -990:
        return "nm" #no truth particle match

    if bh_ancestors.size != 0 and ch_ancestors.size != 0:
        bh_parent = bh_ancestors[0]
        ch_parent = ch_ancestors[0]
        bh_parent_dv = particle_dict[bh_parent[0]].get_dv()
        ch_parent_dv = particle_dict[ch_parent[0]].get_dv()
        if ch_parent[1] < bh_parent[1] and np.linalg.norm(bh_parent_dv-ch_parent_dv) <= threshold_dist and abs(bh_parent[1]-ch_parent[1]) <= threshold_level and np.linalg.norm(ch_parent_dv-track_vertex) <= threshold_dist and ch_parent[1] <= threshold_level:
            return "btoc" #b->c

    if bh_ancestors.size != 0 and (ch_ancestors.size == 0 or bh_ancestors[0,1] < ch_ancestors[0,1]):
        bh_parent = bh_ancestors[0]
        parent_dv = particle_dict[bh_parent[0]].get_dv()
        if np.linalg.norm(parent_dv-track_vertex) <= threshold_dist and bh_parent[1] <= threshold_level:
            return "b" #b

    if ch_ancestors.size != 0 and (bh_ancestors.size == 0 or bh_ancestors[0,1] > ch_ancestors[0,1]):
        ch_parent = ch_ancestors[0]
        parent_dv = particle_dict[ch_parent[0]].get_dv()
        if np.linalg.norm(parent_dv-track_vertex) <= threshold_dist and ch_parent[1] <= threshold_level:
            return "c" #prompt c
    
    return "o" #other decay


def main(argv):
    gROOT.SetBatch(True)

    #ntuple = TFile('/global/homes/t/toyamaza/workdir/ctag/data/ntuples/v10/output_WpHbb_0.root')    
    tree = ntuple.Get("bTag_AntiKt4EMPFlowJets_BTagging201903")

    #association criteria are being the direct descendant (max one level removed) of a HF hadron that spawned a track
    hist_no_char_b = TH1D("no_char_b", "Number of charged particle children per vertex", 10, 0, 10)
    hist_no_char_c = TH1D("no_char_c", "Number of charged particle children per vertex", 10, 0, 10)
    hist_no_char_btoc = TH1D("no_char_btoc", "Number of charged particle children per vertex", 10, 0, 10)
    
    hist_fl_len_btoc = TH1D("fl_len_btoc", "Distance between bH and track vertices in bH->cH event", 20, 0, 50)

    hist_trk_pt_b = TH1D("trk_pt_b", "Track PT", 50, 0, 30000)
    hist_trk_pt_c = TH1D("trk_pt_c", "Track PT", 50, 0, 30000)
    hist_trk_pt_o = TH1D("trk_pt_o", "Track PT", 50, 0, 30000)
    hist_trk_pt_nm = TH1D("trk_pt_nm", "Track PT", 50, 0, 30000)
    hist_trk_pt_btoc = TH1D("trk_pt_btoc", "Track PT", 50, 0, 30000)

    hist_trk_eta_b = TH1D("trk_eta_b", "Track eta", 20, -5, 5)
    hist_trk_eta_c = TH1D("trk_eta_c", "Track eta", 20, -5, 5)
    hist_trk_eta_o = TH1D("trk_eta_o", "Track eta", 20, -5, 5)
    hist_trk_eta_nm = TH1D("trk_eta_nm", "Track eta", 20, -5, 5)
    hist_trk_eta_btoc = TH1D("trk_eta_btoc", "Track eta", 20, -5, 5)

    hist_trk_phi_b = TH1D("trk_phi_b", "Track phi", 20, -math.pi, math.pi)
    hist_trk_phi_c = TH1D("trk_phi_c", "Track phi", 20, -math.pi, math.pi)
    hist_trk_phi_o = TH1D("trk_phi_o", "Track phi", 20, -math.pi, math.pi)
    hist_trk_phi_nm = TH1D("trk_phi_nm", "Track phi", 20, -math.pi, math.pi)
    hist_trk_phi_btoc = TH1D("trk_phi_btoc", "Track phi", 20, -math.pi, math.pi)

    #association criteria are being the most recent HF antecedant of a track particle (not necessarily only one level removed)
    hist_no_trk_b = TH1D("no_trk_b", "Number of associated tracks per vertex", 10, 0, 10)
    hist_no_trk_c = TH1D("no_trk_c", "Number of associated tracks per vertex", 10, 0, 10)
    hist_no_trk_btoc = TH1D("no_trk_btoc", "Number of associated tracks per vertex", 10, 0, 10)

    hist_frac_trk_b = TH1D("frac_trk_b", "Fraction of tracks per jet", 10, 0, 1)
    hist_frac_trk_c = TH1D("frac_trk_c", "Fraction of tracks per jet", 10, 0, 1)
    hist_frac_trk_o = TH1D("frac_trk_o", "Fraction of tracks per jet", 10, 0, 1)
    hist_frac_trk_nm = TH1D("frac_trk_nm", "Fraction of tracks per jet", 10, 0, 1)
    hist_frac_trk_btoc = TH1D("frac_trk_btoc", "Fraction of tracks per jet", 10, 0, 1)

    hist_trk_pv_d0_rej = TH1D("trk_pv_d0_rej", "d0 of rejected tracks", 100, -100, 100)
    hist_trk_pv_d0_acc = TH1D("trk_pv_d0_acc", "d0 of accepted tracks", 100, -100, 100)
    hist_trk_nm_d0_rej = TH1D("trk_nm_d0_rej", "d0 of rejected tracks", 100, -100, 100)
    hist_trk_nm_d0_acc = TH1D("trk_nm_d0_acc", "d0 of accepted tracks", 100, -100, 100)
    hist_trk_o_d0_rej = TH1D("trk_o_d0_rej", "d0 of rejected tracks", 100, -100, 100)
    hist_trk_o_d0_acc = TH1D("trk_o_d0_acc", "d0 of accepted tracks", 100, -100, 100)

    hist_trk_pv_z0_rej = TH1D("trk_pv_z0_rej", "z0 of rejected tracks", 100, -100, 100)
    hist_trk_pv_z0_acc = TH1D("trk_pv_z0_acc", "z0 of accepted tracks", 100, -100, 100)
    hist_trk_nm_z0_rej = TH1D("trk_nm_z0_rej", "z0 of rejected tracks", 100, -100, 100)
    hist_trk_nm_z0_acc = TH1D("trk_nm_z0_acc", "z0 of accepted tracks", 100, -100, 100)
    hist_trk_o_z0_rej = TH1D("trk_nm_z0_rej", "z0 of rejected tracks", 100, -100, 100)
    hist_trk_o_z0_acc = TH1D("trk_nm_z0_acc", "z0 of accepted tracks", 100, -100, 100)

    for ientry,entry in enumerate(tree):

        njets = entry.njets
        print("")
        print("Event {}, Number of Jets: {}".format(ientry, njets))
        
        particle_dict = {}

        #build particle dictionary
        nTruth = entry.truth_pdgId.size()
        print("Number of truth particles: %d"%(nTruth))
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

        primary_vertex = np.array([entry.truth_PVx, entry.truth_PVy, entry.truth_PVz])

        #build track dictionary for each jet
        for i in range( njets ):
            track_dict = {}
            jet_pt  = entry.jet_pt[i]
            jet_eta = entry.jet_eta[i]
            jet_phi = entry.jet_phi[i]
            jet_m   = entry.jet_m[i]
            jet_label = entry.jet_LabDr_HadF[i]

            nTrack =  entry.jet_trk_pt[i].size()
            jet_cut_trk = 0
            for j in range (nTrack):

                trk_vertex = np.array([entry.jet_trk_vtx_X[i][j], entry.jet_trk_vtx_Y[i][j], entry.jet_trk_vtx_Z[i][j]])

                if not remove_pv or not entry.jet_trk_isPV_reco[i][j]:
                    trk_pt = entry.jet_trk_pt[i][j]
                    trk_eta = entry.jet_trk_eta[i][j]
                    trk_phi = entry.jet_trk_phi[i][j]
                    trk_pdgId = entry.jet_trk_pdg_id[i][j]
                    trk_barcode = entry.jet_trk_barcode[i][j]

                    track_dict[trk_barcode] = truth_track(trk_barcode, trk_pdgId, trk_vertex, trk_pt, trk_eta, trk_phi)

                    if trk_vertex[0] < -990:
                        hist_trk_nm_d0_acc.Fill(entry.jet_trk_d0[i][j])
                        hist_trk_nm_z0_acc.Fill(entry.jet_trk_z0[i][j])
                    elif np.linalg.norm(trk_vertex-primary_vertex) < 1e-4:
                        hist_trk_pv_d0_acc.Fill(entry.jet_trk_d0[i][j])
                        hist_trk_pv_z0_acc.Fill(entry.jet_trk_z0[i][j])
                    else:
                        hist_trk_o_d0_acc.Fill(entry.jet_trk_d0[i][j])
                        hist_trk_o_z0_acc.Fill(entry.jet_trk_z0[i][j])

                else:
                    jet_cut_trk += 1

                    if trk_vertex[0] < -990:
                        hist_trk_nm_d0_rej.Fill(entry.jet_trk_d0[i][j])
                        hist_trk_nm_z0_rej.Fill(entry.jet_trk_z0[i][j])
                    elif np.linalg.norm(trk_vertex-primary_vertex) < 1e-4:
                        hist_trk_pv_d0_rej.Fill(entry.jet_trk_d0[i][j])
                        hist_trk_pv_z0_rej.Fill(entry.jet_trk_z0[i][j])
                    else:
                        hist_trk_o_d0_rej.Fill(entry.jet_trk_d0[i][j])
                        hist_trk_o_z0_rej.Fill(entry.jet_trk_z0[i][j])


            jet_npart_trk = 0 #count tracks not associated with any particles
            for t_barcode in track_dict:
                track = track_dict[t_barcode]

                #don't process tracks that don't have associated truth particles
                if track.get_pdgid() == -999:
                    jet_npart_trk += 1
                    continue

                #get ancestors of track particle
                track_particle = particle_dict[t_barcode]
                ancestors = np.array([])
                ancestors = get_ancestors(track_particle, particle_dict, ancestors, 0)
                ancestors = np.reshape(ancestors, (int(len(ancestors)/2),2)) #reshape into array of [barcode, level] elements
                ancestors = ancestors[np.argsort(ancestors[:,1])] #sort based on level
                _, indices = np.unique(ancestors[:,0],return_index=True)
                ancestors = np.array([ancestors[index] for index in sorted(indices)]) #only keep unique barcodes at the lowest level they appear

                #check ancestor list for bH and cH particles
                for ancestor in ancestors:
                    a_barcode = ancestor[0]
                    a_level = ancestor[1]
                    a_particle = particle_dict[a_barcode]
                    a_id = id_particle(a_particle.get_pdgid())

                    #keep track of bH and cH association of each track
                    if a_id == 'ch':
                        track_dict[t_barcode].add_ch_ancestor(np.array([a_barcode, a_level]))

                    elif a_id == 'bh':
                        track_dict[t_barcode].add_bh_ancestor(np.array([a_barcode, a_level]))

            #go through each track again to calculate relevant quantities for plotting
            jet_bh_list = np.array([]) #list that stores all bH particles in jet
            jet_ch_list = np.array([])
            jet_trk_btoc_frac = jet_trk_ch_frac = jet_trk_bh_frac = jet_trk_o_frac = jet_trk_nm_frac = 0
            bh_parent_list = np.array([]) #list that stores all direct bH parent particles in jet
            ch_parent_list = np.array([])
            btoc_parent_list = np.array([])

            for t_barcode in track_dict:
                #create list of all bH and cH particles in one jet
                t_bh_list = track_dict[t_barcode].get_bh_ancestors()
                t_ch_list = track_dict[t_barcode].get_ch_ancestors()
                if t_bh_list.size != 0: jet_bh_list = np.append(jet_bh_list, t_bh_list[:,0])
                if t_ch_list.size != 0: jet_ch_list = np.append(jet_ch_list, t_ch_list[:,0])
                
                #classify tracks and save classification
                t_class = classify_track(t_barcode, particle_dict, track_dict)
                track_dict[t_barcode].set_classification(t_class)
                
                if t_class == 'nm':
                    jet_trk_nm_frac += 1/len(track_dict)
                    hist_trk_pt_nm.Fill(track_dict[t_barcode].get_pt())
                    hist_trk_eta_nm.Fill(track_dict[t_barcode].get_eta())
                    hist_trk_phi_nm.Fill(track_dict[t_barcode].get_phi())
                elif t_class == 'b':
                    jet_trk_bh_frac += 1/len(track_dict)
                    hist_trk_pt_b.Fill(track_dict[t_barcode].get_pt())
                    hist_trk_eta_b.Fill(track_dict[t_barcode].get_eta())
                    hist_trk_phi_b.Fill(track_dict[t_barcode].get_phi())
                    bh_parent_list = np.append(bh_parent_list, t_bh_list[0,0])
                elif t_class == 'c':
                    jet_trk_ch_frac += 1/len(track_dict)
                    hist_trk_pt_c.Fill(track_dict[t_barcode].get_pt())
                    hist_trk_eta_c.Fill(track_dict[t_barcode].get_eta())
                    hist_trk_phi_c.Fill(track_dict[t_barcode].get_phi())
                    ch_parent_list = np.append(ch_parent_list, t_ch_list[0,0])
                elif t_class == 'btoc':
                    jet_trk_btoc_frac += 1/len(track_dict)
                    hist_trk_pt_btoc.Fill(track_dict[t_barcode].get_pt())
                    hist_trk_eta_btoc.Fill(track_dict[t_barcode].get_eta())
                    hist_trk_phi_btoc.Fill(track_dict[t_barcode].get_phi())
                    btoc_parent_list = np.append(btoc_parent_list, t_ch_list[0,0])
                    bH_parent = particle_dict[t_bh_list[0,0]]
                    cH_parent = particle_dict[t_ch_list[0,0]]
                    hist_fl_len_btoc.Fill(np.linalg.norm(bH_parent.get_dv()-cH_parent.get_dv()))
                else:
                    jet_trk_o_frac += 1/len(track_dict)

            #fill histograms
            if len(track_dict) != 0:
                hist_frac_trk_b.Fill(jet_trk_bh_frac)
                hist_frac_trk_c.Fill(jet_trk_ch_frac)
                hist_frac_trk_o.Fill(jet_trk_o_frac)
                hist_frac_trk_btoc.Fill(jet_trk_btoc_frac)
                hist_frac_trk_nm.Fill(jet_trk_nm_frac)

                bh_parent_list, bh_unique = np.unique(bh_parent_list, return_counts=True)
                for incidence in bh_unique:
                    hist_no_trk_b.Fill(incidence)
                for parent in bh_parent_list:
                    bh_children = particle_dict[parent].get_children()
                    bh_charged = 0
                    for barcode in bh_children:
                        if particle_dict[int(barcode)].is_charged():
                            bh_charged += 1
                    hist_no_char_b.Fill(bh_charged)

                ch_parent_list, ch_unique = np.unique(ch_parent_list, return_counts=True)
                for incidence in ch_unique:
                    hist_no_trk_c.Fill(incidence)
                for parent in ch_parent_list:
                    ch_children = particle_dict[parent].get_children()
                    ch_charged = 0
                    for barcode in ch_children:
                        if particle_dict[int(barcode)].is_charged():
                            ch_charged += 1
                    hist_no_char_c.Fill(ch_charged)
               
                btoc_parent_list, btoc_unique = np.unique(btoc_parent_list, return_counts=True)
                for incidence in btoc_unique:
                    hist_no_trk_btoc.Fill(incidence)
                for parent in btoc_parent_list:
                    btoc_children = particle_dict[parent].get_children()
                    btoc_charged = 0
                    for barcode in btoc_children:
                        if particle_dict[int(barcode)].is_charged():
                            btoc_charged += 1
                    hist_no_char_btoc.Fill(btoc_charged)

            jet_nbh = np.size(np.unique(jet_bh_list))
            jet_nch = np.size(np.unique(jet_ch_list))

        if ientry>=maxentries-1: break # the first event only
        
    canv1 = TCanvas("c1", "c1", 800, 600)
    gPad.SetLogy()
    hist_frac_trk_b.GetXaxis().SetTitle("Track fraction")
    hist_frac_trk_b.GetYaxis().SetTitle("Entries")
    hist_frac_trk_b.SetLineColor(4)
    hist_frac_trk_b.SetFillColorAlpha(4,0.2)
    hist_frac_trk_b.Draw()
    hist_frac_trk_c.SetLineColor(3)
    hist_frac_trk_c.SetFillColorAlpha(3,0.2)
    hist_frac_trk_c.Draw("SAMES")
    hist_frac_trk_btoc.SetLineColor(2)
    hist_frac_trk_btoc.SetFillColorAlpha(2,0.2)
    hist_frac_trk_btoc.Draw("SAMES")
    gPad.Update()
    stats_c = hist_frac_trk_c.FindObject("stats")
    stats_c.SetX1NDC(0.58)
    stats_c.SetX2NDC(0.78)
    stats_b = hist_frac_trk_b.FindObject("stats")
    stats_b.SetX1NDC(0.38)
    stats_b.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_frac_trk_b, "bH", "l")
    legend.AddEntry(hist_frac_trk_c, "prompt cH", "l")
    legend.AddEntry(hist_frac_trk_btoc, "bH->cH", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_frac_trk.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_no_char_b.GetXaxis().SetTitle("Number of particles")
    hist_no_char_b.GetYaxis().SetTitle("Normalized entries")
    hist_no_char_b.SetLineColor(4)
    hist_no_char_b.SetFillColorAlpha(4,0.2)
    hist_no_char_b.Scale(1./hist_no_char_b.Integral(), "width")
    hist_no_char_b.Draw()
    hist_no_char_c.SetLineColor(3)
    hist_no_char_c.SetFillColorAlpha(3,0.2)
    hist_no_char_c.Scale(1./hist_no_char_c.Integral(), "width")
    hist_no_char_c.Draw("SAMES")
    hist_no_char_btoc.SetLineColor(2)
    hist_no_char_btoc.SetFillColorAlpha(2,0.2)
    hist_no_char_btoc.Scale(1./hist_no_char_btoc.Integral(), "width")
    hist_no_char_btoc.Draw("SAMES")
    gPad.Update()
    stats_c = hist_no_char_c.FindObject("stats")
    stats_c.SetX1NDC(0.58)
    stats_c.SetX2NDC(0.78)
    stats_b = hist_no_char_b.FindObject("stats")
    stats_b.SetX1NDC(0.38)
    stats_b.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_no_char_b, "bH", "l")
    legend.AddEntry(hist_no_char_c, "prompt cH", "l")
    legend.AddEntry(hist_no_char_btoc, "bH->cH", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_no_char.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_no_trk_b.GetXaxis().SetTitle("Number of tracks")
    hist_no_trk_b.GetYaxis().SetTitle("Normalized entries")
    hist_no_trk_b.SetLineColor(4)
    hist_no_trk_b.SetFillColorAlpha(4,0.2)
    hist_no_trk_b.Scale(1./hist_no_trk_b.Integral(), "width")
    hist_no_trk_b.Draw()
    hist_no_trk_c.SetLineColor(3)
    hist_no_trk_c.SetFillColorAlpha(3,0.2)
    hist_no_trk_c.Scale(1./hist_no_trk_c.Integral(), "width")
    hist_no_trk_c.Draw("SAMES")
    hist_no_trk_btoc.SetLineColor(2)
    hist_no_trk_btoc.SetFillColorAlpha(2,0.2)
    hist_no_trk_btoc.Scale(1./hist_no_trk_btoc.Integral(), "width")
    hist_no_trk_btoc.Draw("SAMES")
    gPad.Update()
    stats_c = hist_no_trk_c.FindObject("stats")
    stats_c.SetX1NDC(0.58)
    stats_c.SetX2NDC(0.78)
    stats_b = hist_no_trk_b.FindObject("stats")
    stats_b.SetX1NDC(0.38)
    stats_b.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_no_trk_b, "bH", "l")
    legend.AddEntry(hist_no_trk_c, "prompt cH", "l")
    legend.AddEntry(hist_no_trk_btoc, "bH->cH", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_no_trk.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_fl_len_btoc.GetXaxis().SetTitle("Distance [cm]")
    hist_fl_len_btoc.GetYaxis().SetTitle("Entries")
    hist_fl_len_btoc.Draw()
    canv1.SaveAs(savepath+dataname+"_fl_len_btoc.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_pt_b.GetXaxis().SetTitle("pT [MeV]")
    hist_trk_pt_b.GetYaxis().SetTitle("Normalized entries")
    hist_trk_pt_b.SetLineColor(4)
    hist_trk_pt_b.SetFillColorAlpha(4,0.2)
    hist_trk_pt_b.Scale(1./hist_trk_pt_b.Integral(), "width")
    hist_trk_pt_b.Draw("HIST")
    hist_trk_pt_c.SetLineColor(2)
    hist_trk_pt_c.SetFillColorAlpha(2,0.2)
    hist_trk_pt_c.Scale(1./hist_trk_pt_c.Integral(), "width")
    hist_trk_pt_c.Draw("SAMES HIST")
    hist_trk_pt_btoc.SetLineColor(3)
    hist_trk_pt_btoc.SetFillColorAlpha(3,0.2)
    hist_trk_pt_btoc.Scale(1./hist_trk_pt_btoc.Integral(), "width")
    hist_trk_pt_btoc.Draw("SAMES HIST")
    gPad.Update()
    stats_c = hist_trk_pt_c.FindObject("stats")
    stats_c.SetX1NDC(0.58)
    stats_c.SetX2NDC(0.78)
    stats_b = hist_trk_pt_b.FindObject("stats")
    stats_b.SetX1NDC(0.38)
    stats_b.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_pt_b, "bH", "l")
    legend.AddEntry(hist_trk_pt_c, "prompt cH", "l")
    legend.AddEntry(hist_trk_pt_btoc, "bH->cH", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_pt.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_eta_b.GetXaxis().SetTitle("Eta")
    hist_trk_eta_b.GetYaxis().SetTitle("Normalized entries")
    hist_trk_eta_b.SetLineColor(4)
    hist_trk_eta_b.SetFillColorAlpha(4,0.2)
    hist_trk_eta_b.Scale(1./hist_trk_eta_b.Integral(), "width")
    hist_trk_eta_b.Draw("HIST")
    hist_trk_eta_c.SetLineColor(2)
    hist_trk_eta_c.SetFillColorAlpha(2,0.2)
    hist_trk_eta_c.Scale(1./hist_trk_eta_c.Integral(), "width")
    hist_trk_eta_c.Draw("SAMES HIST")
    hist_trk_eta_btoc.SetLineColor(3)
    hist_trk_eta_btoc.SetFillColorAlpha(3,0.2)
    hist_trk_eta_btoc.Scale(1./hist_trk_eta_btoc.Integral(), "width")
    hist_trk_eta_btoc.Draw("SAMES HIST")
    gPad.Update()
    stats_c = hist_trk_eta_c.FindObject("stats")
    stats_c.SetX1NDC(0.58)
    stats_c.SetX2NDC(0.78)
    stats_b = hist_trk_eta_b.FindObject("stats")
    stats_b.SetX1NDC(0.38)
    stats_b.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_eta_b, "bH", "l")
    legend.AddEntry(hist_trk_eta_c, "prompt cH", "l")
    legend.AddEntry(hist_trk_eta_btoc, "bH->cH", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_eta.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_phi_b.GetXaxis().SetTitle("Phi")
    hist_trk_phi_b.GetYaxis().SetTitle("Normalized entries")
    hist_trk_phi_b.SetLineColor(4)
    hist_trk_phi_b.SetFillColorAlpha(4,0.2)
    hist_trk_phi_b.Scale(1./hist_trk_phi_b.Integral(), "width")
    hist_trk_phi_b.Draw("HIST")
    hist_trk_phi_c.SetLineColor(2)
    hist_trk_phi_c.SetFillColorAlpha(2,0.2)
    hist_trk_phi_c.Scale(1./hist_trk_phi_c.Integral(), "width")
    hist_trk_phi_c.Draw("SAMES HIST")
    hist_trk_phi_btoc.SetLineColor(3)
    hist_trk_phi_btoc.SetFillColorAlpha(3,0.2)
    hist_trk_phi_btoc.Scale(1./hist_trk_phi_btoc.Integral(), "width")
    hist_trk_phi_btoc.Draw("SAMES HIST")
    gPad.Update()
    stats_c = hist_trk_phi_c.FindObject("stats")
    stats_c.SetX1NDC(0.58)
    stats_c.SetX2NDC(0.78)
    stats_b = hist_trk_phi_b.FindObject("stats")
    stats_b.SetX1NDC(0.38)
    stats_b.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_phi_b, "bH", "l")
    legend.AddEntry(hist_trk_phi_c, "prompt cH", "l")
    legend.AddEntry(hist_trk_phi_btoc, "bH->cH", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_phi.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_pv_d0_acc.GetXaxis().SetTitle("Distance [cm]")
    hist_trk_pv_d0_acc.GetYaxis().SetTitle("Normalized ntries")
    hist_trk_pv_d0_acc.SetLineColor(4)
    hist_trk_pv_d0_acc.SetFillColorAlpha(4,0.2)
    hist_trk_pv_d0_acc.Scale(1./hist_trk_pv_d0_acc.Integral(), "width")
    hist_trk_pv_d0_acc.Draw("HIST")
    hist_trk_o_d0_acc.SetLineColor(2)
    hist_trk_o_d0_acc.SetFillColorAlpha(2,0.2)
    hist_trk_o_d0_acc.Scale(1./hist_trk_o_d0_acc.Integral(), "width")
    hist_trk_o_d0_acc.Draw("SAMES HIST")
    hist_trk_nm_d0_acc.SetLineColor(3)
    hist_trk_nm_d0_acc.SetFillColorAlpha(3,0.2)
    hist_trk_nm_d0_acc.Scale(1./hist_trk_nm_d0_acc.Integral(), "width")
    hist_trk_nm_d0_acc.Draw("SAMES HIST")
    gPad.Update()
    stats_o = hist_trk_o_d0_acc.FindObject("stats")
    stats_o.SetX1NDC(0.58)
    stats_o.SetX2NDC(0.78)
    stats_pv = hist_trk_nm_d0_acc.FindObject("stats")
    stats_pv.SetX1NDC(0.38)
    stats_pv.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_pv_d0_acc, "pv associated", "l")
    legend.AddEntry(hist_trk_o_d0_acc, "non pv associated", "l")
    legend.AddEntry(hist_trk_nm_d0_acc, "no match", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_d0_acc.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_pv_z0_acc.GetXaxis().SetTitle("Distance [cm]")
    hist_trk_pv_z0_acc.GetYaxis().SetTitle("Normalized entries")
    hist_trk_pv_z0_acc.SetLineColor(4)
    hist_trk_pv_z0_acc.SetFillColorAlpha(4,0.2)
    hist_trk_pv_z0_acc.Scale(1./hist_trk_pv_z0_acc.Integral(), "width")
    hist_trk_pv_z0_acc.Draw("HIST")
    hist_trk_o_z0_acc.SetLineColor(2)
    hist_trk_o_z0_acc.SetFillColorAlpha(2,0.2)
    hist_trk_o_z0_acc.Scale(1./hist_trk_o_z0_acc.Integral(), "width")
    hist_trk_o_z0_acc.Draw("SAMES HIST")
    hist_trk_nm_z0_acc.SetLineColor(3)
    hist_trk_nm_z0_acc.SetFillColorAlpha(3,0.2)
    hist_trk_nm_z0_acc.Scale(1./hist_trk_nm_z0_acc.Integral(), "width")
    hist_trk_nm_z0_acc.Draw("SAMES HIST")
    gPad.Update()
    stats_o = hist_trk_o_z0_acc.FindObject("stats")
    stats_o.SetX1NDC(0.58)
    stats_o.SetX2NDC(0.78)
    stats_pv = hist_trk_nm_z0_acc.FindObject("stats")
    stats_pv.SetX1NDC(0.38)
    stats_pv.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_pv_z0_acc, "pv associated", "l")
    legend.AddEntry(hist_trk_o_z0_acc, "non pv associated", "l")
    legend.AddEntry(hist_trk_nm_z0_acc, "no match", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_z0_acc.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_pv_d0_rej.GetXaxis().SetTitle("Distance [cm]")
    hist_trk_pv_d0_rej.GetYaxis().SetTitle("Normalized entries")
    hist_trk_pv_d0_rej.SetLineColor(4)
    hist_trk_pv_d0_rej.SetFillColorAlpha(4,0.2)
    hist_trk_pv_d0_rej.Scale(1./hist_trk_pv_d0_rej.Integral(), "width")
    hist_trk_pv_d0_rej.Draw("HIST")
    hist_trk_o_d0_rej.SetLineColor(2)
    hist_trk_o_d0_rej.SetFillColorAlpha(2,0.2)
    hist_trk_o_d0_rej.Scale(1./hist_trk_o_d0_rej.Integral(), "width")
    hist_trk_o_d0_rej.Draw("SAMES HIST")
    hist_trk_nm_d0_rej.SetLineColor(3)
    hist_trk_nm_d0_rej.SetFillColorAlpha(3,0.2)
    hist_trk_nm_d0_rej.Scale(1./hist_trk_nm_d0_rej.Integral(), "width")
    hist_trk_nm_d0_rej.Draw("SAMES HIST")
    gPad.Update()
    stats_o = hist_trk_o_d0_rej.FindObject("stats")
    stats_o.SetX1NDC(0.58)
    stats_o.SetX2NDC(0.78)
    stats_pv = hist_trk_nm_d0_rej.FindObject("stats")
    stats_pv.SetX1NDC(0.38)
    stats_pv.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_pv_d0_rej, "pv associated", "l")
    legend.AddEntry(hist_trk_o_d0_rej, "non pv associated", "l")
    legend.AddEntry(hist_trk_nm_d0_rej, "no match", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_d0_rej.png")
    gPad.Clear()
    canv1.Clear()

    gPad.SetLogy()
    hist_trk_pv_z0_rej.GetXaxis().SetTitle("Distance [cm]")
    hist_trk_pv_z0_rej.GetYaxis().SetTitle("Normalized entries")
    hist_trk_pv_z0_rej.SetLineColor(4)
    hist_trk_pv_z0_rej.SetFillColorAlpha(4,0.2)
    hist_trk_pv_z0_rej.Scale(1./hist_trk_pv_z0_rej.Integral(), "width")
    hist_trk_pv_z0_rej.Draw("HIST")
    hist_trk_o_z0_rej.SetLineColor(2)
    hist_trk_o_z0_rej.SetFillColorAlpha(2,0.2)
    hist_trk_o_z0_rej.Scale(1./hist_trk_o_z0_rej.Integral(), "width")
    hist_trk_o_z0_rej.Draw("SAMES HIST")
    hist_trk_nm_z0_rej.SetLineColor(3)
    hist_trk_nm_z0_rej.SetFillColorAlpha(3,0.2)
    hist_trk_nm_z0_rej.Scale(1./hist_trk_nm_z0_rej.Integral(), "width")
    hist_trk_nm_z0_rej.Draw("SAMES HIST")
    gPad.Update()
    stats_o = hist_trk_o_z0_rej.FindObject("stats")
    stats_o.SetX1NDC(0.58)
    stats_o.SetX2NDC(0.78)
    stats_pv = hist_trk_nm_z0_rej.FindObject("stats")
    stats_pv.SetX1NDC(0.38)
    stats_pv.SetX2NDC(0.58)
    legend = TLegend(0.78,0.65,0.98,0.75)
    legend.AddEntry(hist_trk_pv_z0_rej, "pv associated", "l")
    legend.AddEntry(hist_trk_o_z0_rej, "non pv associated", "l")
    legend.AddEntry(hist_trk_nm_z0_rej, "no match", "l")
    legend.Draw("SAME")
    canv1.SaveAs(savepath+dataname+"_trk_z0_rej.png")
    gPad.Clear()
    canv1.Clear()


if __name__ == '__main__':
    main(sys.argv)
