#!/usr/bin/env python

import os,sys,math,ROOT,glob
import numpy as np
from ROOT import TFile, TH1D, gROOT, TCanvas, gPad

maxentries = 100

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

    def print_particle(self,level):
        prefix = "--"*level + ">"
        print("{} PDG ID: {}, Barcode: {}, Parents: {}, Children: {}, PV: {}, DV: {}".format(prefix, self.pdgid, self.barcode, self.parents, self.children, self.pv, self.dv))


class truth_track():
    def __init__(self, barcode, pdgid, vertex):
        self.barcode = barcode
        self.pdgid = pdgid
        self.vertex = vertex

    def get_pdgid(self):
        return self.parent_pdgid

    def get_barcode(self):
        return self.barcode

    def print_track(self,level):
        prefix = "xx"*level + ">"
        print("{} PDG ID: {}, Barcode: {}, Vertex: {}".format(prefix, self.pdgid, self.barcode, self.vertex))


def print_tree(particle, particle_dict, track_dict, level):
    barcode = particle.get_barcode()
    pdgid = particle.get_pdgid()
    condition = True #(id_particle(pdgid) == 'ch') or (id_particle(pdgid) == 'bh')

    #print information
    if condition:
        if barcode in track_dict:
            track_dict[barcode].print_track(level)
        particle.print_particle(level)

    #loop through children recursively
    if level:
        for child in particle.get_children():
            print_tree(particle_dict[child], particle_dict, track_dict, level+1)


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


def check_children(particle, particle_dict, particle_list): #particle_list stores barcodes to avoid double counting
    barcode = particle.get_barcode()
    pdgid = particle.get_pdgid()
    condition = (id_particle(pdgid) == 'ch')

    if condition:
        particle_list = np.append(particle_list, barcode)
        
    for child in particle.get_children():
        particle_list = check_children(particle_dict[child], particle_dict, particle_list)
    
    return particle_list


def main(argv):
    gROOT.SetBatch(True)

    ntuple = TFile('/global/homes/t/toyamaza/workdir/ctag/data/ntuples/v10/output_WpHbb_0.root')    
    tree = ntuple.Get("bTag_AntiKt4EMPFlowJets_BTagging201903")

    hist_fl_len = TH1D("fl_len", "HF hadron flight length", 20, 0, 50)
    hist_num_trk = TH1D("num_trk", "Number of tracks associated with HF vertex", 10, 0, 10)
    hist_num_char = TH1D("num_char", "Number of charged particles associated with HF vertex", 20, 0, 20)
    hist_num_hf = TH1D("num_hf", "Number of HF hadrons per event", 10, 0, 10)

    hist_fl_len_b = TH1D("fl_len_b", "b hadron flight length", 20, 0, 50)
    hist_num_trk_b = TH1D("num_trk_b", "Number of tracks associated with b vertex", 10, 0, 10)
    hist_num_char_b = TH1D("num_char_b", "Number of charged particles associated with b vertex", 20, 0, 20)
    hist_num_b = TH1D("num_b", "Number of b hadrons per event", 10, 0, 10)

    hist_fl_len_c = TH1D("fl_len_c", "c hadron flight length", 20, 0, 50)
    hist_num_trk_c = TH1D("num_trk_c", "Number of tracks associated with c vertex", 10, 0, 10)
    hist_num_char_c = TH1D("num_char_c", "Number of charged particles associated with c vertex", 20, 0, 20)
    hist_num_c = TH1D("num_c", "Number of c hadrons per event", 10, 0, 10)

    hist_btoc_dist = TH1D("btoc_dist", "distance between bH and cH production vertices", 20, 0, 50)

    total_legit = 0 ####
    total_trk = 0

    total_particles = 0
    total_ch = 0
    total_bh = 0
    total_ch_wtrk = 0
    total_bh_wtrk = 0
    total_btoc = 0 #total b hadrons decaying to c hadrons
    for ientry,entry in enumerate(tree):

        njets =  entry.njets
        print("")
        print("Event {}, Number of Jets: {}".format(ientry, njets))
        
        track_dict = {}
        particle_dict = {}

        #build track dictionary
        for i in range( njets ):
            jet_pt  = entry.jet_pt[i]
            jet_eta = entry.jet_eta[i]
            jet_phi = entry.jet_phi[i]
            jet_m   = entry.jet_m[i]
            jet_label = entry.jet_LabDr_HadF[i]
            
            nTrack =  entry.jet_trk_pt[i].size()
            for j in range ( nTrack ):
                trk_pt = entry.jet_trk_pt[i][j]
                trk_eta = entry.jet_trk_eta[i][j]
                trk_phi = entry.jet_trk_phi[i][j]
                trk_pdgId = entry.jet_trk_pdg_id[i][j]
                trk_barcode = entry.jet_trk_barcode[i][j]
                trk_vertex_x = entry.jet_trk_vtx_X[i][j]
                trk_vertex_y = entry.jet_trk_vtx_Y[i][j]
                trk_vertex_z = entry.jet_trk_vtx_Z[i][j]

                trk_vertex = np.array([trk_vertex_x, trk_vertex_y, trk_vertex_z])
                track_dict[trk_barcode] = truth_track(trk_barcode, trk_pdgId, trk_vertex)

                total_trk += 1
                if trk_vertex[0] > -990:
                    total_legit += 1

        #build particle dictionary
        nTruth = entry.truth_pdgId.size()
        print("Number of truth particles: %d"%(nTruth))
        for i in range ( nTruth ):
            truth_pdgId = entry.truth_pdgId[i]
            truth_barcode = entry.truth_barcode[i]
            truth_status = entry.truth_status[i]
            truth_pvx = entry.truth_pvtx_x[i]
            truth_pvy = entry.truth_pvtx_y[i]
            truth_pvz = entry.truth_pvtx_z[i]
            truth_dvx = entry.truth_dvtx_x[i]
            truth_dvy = entry.truth_dvtx_y[i]
            truth_dvz = entry.truth_dvtx_z[i]
            truth_px = entry.truth_px[i]
            truth_py = entry.truth_py[i]
            truth_pz = entry.truth_pz[i]
            truth_charged = entry.truth_isCharged[i]

            truth_pv = np.array([truth_pvx, truth_pvy, truth_pvz])
            truth_dv = np.array([truth_dvx, truth_dvy, truth_dvz])
            truth_p = np.array([truth_px, truth_py, truth_pz])

            truth_nParent = entry.truth_parent_pdgId[i].size()
            truth_nChild = entry.truth_child_pdgId[i].size()
           
            output = "barcode =%8d, pdgId =%9d, status =%3d, nParent= %d, nChild=%d : "%(truth_barcode,truth_pdgId, truth_status,truth_nParent,truth_nChild)

            child_pdgId_array = np.zeros(truth_nChild)
            child_barcode_array = np.zeros(truth_nChild)
            for j in range (truth_nChild):
                child_pdgId = entry.truth_child_pdgId[i][j]
                child_pdgId_array[j] = child_pdgId
                child_barcode = entry.truth_child_barcode[i][j]
                child_barcode_array[j] = child_barcode
                output+= "  %d (barcode:%d)"%(child_pdgId,child_barcode)
            
            parent_pdgId_array = np.zeros(truth_nParent)
            parent_barcode_array = np.zeros(truth_nParent)
            for j in range (truth_nParent):
                parent_pdgId = entry.truth_parent_pdgId[i][j]
                parent_pdgId_array[j] = parent_pdgId
                parent_barcode = entry.truth_parent_barcode[i][j]
                parent_barcode_array[j] = parent_barcode
                output+= "  %d (barcode:%d)"%(parent_pdgId,parent_barcode)

            particle_dict[truth_barcode] = truth_particle(truth_barcode, truth_pdgId, truth_pv, truth_dv, truth_charged, truth_p, parent_barcode_array, child_barcode_array)
            
            #print(output)
        
        #loop through track_dict
        #for barcode in track_dict:
            #if barcode in particle_dict: #check if track has truth particle
                #particle = particle_dict[barcode]

        #loop through particle_dict
        event_bh = 0
        event_ch = 0
        event_hf = 0
        total_particles += len(particle_dict)
        btoc_list = np.array([])
        for barcode in particle_dict:
            particle = particle_dict[barcode]
            pdgid = particle.get_pdgid()
            pid = id_particle(pdgid)
            
            if ((pid == 'ch') or (pid == 'bh')):

                event_hf += 1
                pv = particle.get_pv()
                dv = particle.get_dv()
                if dv[0] >= -990. and pv[0] >= -990.: #check that particle has decay and production vertex

                    tracks = 0 #number of spawned tracks
                    charged = 0 #number of charged children
                    for c_barcode in particle.get_children():
                        child = particle_dict[c_barcode]
                        cpid = id_particle(child.get_pdgid())
                        if c_barcode in track_dict:
                            tracks += 1
                        if child.is_charged():
                            charged += 1
                    
                    hist_num_trk.Fill(tracks)
                    hist_fl_len.Fill(np.linalg.norm(pv-dv))
                    hist_num_char.Fill(charged)

                    if pid == 'ch':
                        total_ch += 1
                        event_ch += 1
                        if tracks > 0:
                            total_ch_wtrk += 1
                        hist_fl_len_c.Fill(np.linalg.norm(pv-dv))
                        hist_num_trk_c.Fill(tracks)
                        hist_num_char_c.Fill(charged)
                    
                    elif pid == 'bh':
                        total_bh += 1
                        event_bh += 1
                        if tracks > 0:
                            total_bh_wtrk += 1
                        hist_fl_len_b.Fill(np.linalg.norm(pv-dv))
                        hist_num_trk_b.Fill(tracks)
                        hist_num_char_b.Fill(charged)

                        #calculate/print information about b->c events (only line 2 required for plots)
                        btoc_len = np.size(btoc_list)
                        btoc_list = check_children(particle, particle_dict, btoc_list)
                        btoc_new = btoc_list[btoc_len:]
                        btoc_ids = btoc_dist = np.zeros(len(btoc_new))
                        for i in range(len(btoc_new)):
                            btoc_ids[i] = particle_dict[btoc_new[i]].get_pdgid()
                            hist_btoc_dist.Fill(np.linalg.norm(pv-particle_dict[btoc_new[i]].get_pv()))
                        print("bH: {} ({}); cH: {} ({})".format(barcode, pdgid, btoc_new, btoc_ids))
                        
            #print full tree
            #if len(particle.get_parent()) == 0:
                #print_tree(particle, particle_dict, track_dict, 1)
       
        total_btoc += np.size(np.unique(btoc_list))
        hist_num_hf.Fill(event_hf)
        hist_num_b.Fill(event_bh)
        hist_num_c.Fill(event_ch)

        if ientry>=maxentries-1: break # the first event only
        
    print(total_trk, total_legit)
    print("Total: {}, cH: {}, bH: {}, bH->cH: {}, cH (w track): {}, bH (w track): {}".format(total_particles, total_ch, total_bh, total_btoc, total_ch_wtrk, total_bh_wtrk))
    
    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_fl_len.GetXaxis().SetTitle("Distance [?]")
    hist_fl_len.GetYaxis().SetTitle("Entries")
    hist_fl_len.Draw()
    canv1.SaveAs("fl_len.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_trk.GetXaxis().SetTitle("Number of tracks")
    hist_num_trk.GetYaxis().SetTitle("Entries")
    hist_num_trk.Draw()
    canv1.SaveAs("num_trk.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_char.GetXaxis().SetTitle("Number of charged particles")
    hist_num_char.GetYaxis().SetTitle("Entries")
    hist_num_char.Draw()
    canv1.SaveAs("num_char.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_hf.GetXaxis().SetTitle("Number of HF hadrons")
    hist_num_hf.GetYaxis().SetTitle("Entries")
    hist_num_hf.Draw()
    canv1.SaveAs("num_hf.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_fl_len_c.GetXaxis().SetTitle("Distance [?]")
    hist_fl_len_c.GetYaxis().SetTitle("Entries")
    hist_fl_len_c.Draw()
    canv1.SaveAs("fl_len_c.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_trk_c.GetXaxis().SetTitle("Number of tracks")
    hist_num_trk_c.GetYaxis().SetTitle("Entries")
    hist_num_trk_c.Draw()
    canv1.SaveAs("num_trk_c.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_char_c.GetXaxis().SetTitle("Number of charged particles")
    hist_num_char_c.GetYaxis().SetTitle("Entries")
    hist_num_char_c.Draw()
    canv1.SaveAs("num_char_c.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_c.GetXaxis().SetTitle("Number of c hadrons")
    hist_num_c.GetYaxis().SetTitle("Entries")
    hist_num_c.Draw()
    canv1.SaveAs("num_c.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_fl_len_b.GetXaxis().SetTitle("Distance [?]")
    hist_fl_len_b.GetYaxis().SetTitle("Entries")
    hist_fl_len_b.Draw()
    canv1.SaveAs("fl_len_b.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_trk_b.GetXaxis().SetTitle("Number of tracks")
    hist_num_trk_b.GetYaxis().SetTitle("Log(Entries)")
    hist_num_trk_b.Draw()
    canv1.SaveAs("num_trk_b.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_char_b.GetXaxis().SetTitle("Number of charged particles")
    hist_num_char_b.GetYaxis().SetTitle("Entries")
    hist_num_char_b.Draw()
    canv1.SaveAs("num_char_b.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_num_b.GetXaxis().SetTitle("Number of b hadrons")
    hist_num_b.GetYaxis().SetTitle("Entries")
    hist_num_b.Draw()
    canv1.SaveAs("num_b.png")

    canv1 = TCanvas("c1","c1", 800, 600)
    gPad.SetLogy()
    hist_btoc_dist.GetXaxis().SetTitle("Distance [?]")
    hist_btoc_dist.GetYaxis().SetTitle("Entries")
    hist_btoc_dist.Draw()
    canv1.SaveAs("btoc_dist.png")

if __name__ == '__main__':
    main(sys.argv)
