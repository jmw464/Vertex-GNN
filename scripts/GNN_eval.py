import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,time
import numpy as np
import argparse
import ROOT
from ROOT import gROOT, gStyle, TFile, TH1D, TLegend, TCanvas
import matplotlib.pyplot as plt


#evaluate confusion matrix for binary case
def evaluate_confusion_bin(true, pred):

    cm = np.zeros((2,2),dtype=int)
    cm[1,1] = np.sum((true[:,0] == 1) & (pred[:,0] == 1)) #true positive - actually true, marked as true
    cm[0,0] = np.sum((true[:,0] == 0) & (pred[:,0] == 0)) #true negative - actually false, marked as false
    cm[0,1] = np.sum((true[:,0] == 0) & (pred[:,0] == 1)) #false positive - actually false, marked as true
    cm[1,0] = np.sum((true[:,0] == 1) & (pred[:,0] == 0)) #false negative - actually true, marked as false
    
    return cm


#evaluate confusion matrix for multi-class case
def evaluate_confusion_mult(true, pred):

    cm = np.zeros((5,5),dtype=int)
    for i in range(5):
        for j in range(5):
            cm[i,j] = np.sum((true[:,0] == i) & (pred[:] == j))
    
    return cm


#get list of events GNN performs poorly on
def evaluate_jet(graph, hist_list, bad_events, multi_class, bin_threshold, mult_threshold):

    node_info = graph.ndata['info'].cpu().detach().numpy().astype(int)

    #store recall for each class
    if not multi_class:
        r_array = np.zeros(2)
        pred = th.round(graph.edata['pred']).cpu().detach().numpy().astype(int)
        true = graph.edata['bin_labels'].cpu().detach().numpy().astype(int)
        cm = evaluate_confusion_bin(true, pred)
        r_threshold = bin_threshold
    else:
        r_array = np.zeros(5)
        pred = th.argmax(graph.edata['pred']).cpu().detach().numpy().astype(int)
        true = graph.edata['mult_labels'].cpu().detach().numpy().astype(int)
        cm = evaluate_confusion_mult(true, pred)
        r_threshold = mult_threshold
            
    #fill histograms and r_array to determine bad events
    for j in range(cm.shape[0]):
        if np.sum(cm[j,:]) != 0:
            r_array[j] = cm[j,j]/np.sum(cm[j,:])
            hist_list[j].Fill(r_array[j])
        else:
            r_array[j] = -1

    for j in range(cm.shape[0]):
        if r_array[j] < r_threshold[j] and r_array[j] >= 0:
            bad_events = np.append(bad_events, [[node_info[0,0], node_info[0,1], node_info[0,2]]], axis=0)
            break

    return bad_events


def find_vertices_bin(graph, mode, score_threshold):
    ntracks = graph.number_of_nodes()
    edges = graph.all_edges()
    um_tracks = np.array(range(ntracks))

    if mode == 'truth':
        labels = graph.edata['bin_labels'].cpu().detach().numpy().flatten()
    else:
        labels = graph.edata['pred'].cpu().detach().numpy().flatten()

    vertices = []

    #average out forward and backward edges
    for i in range(ntracks*(ntracks-1)-1):
        labels[i] += labels[i+1]
        labels[i] = labels[i]/2.
    um_labels = labels[::2] #these arrays only contain values of unmatched edges
    um_senders = edges[0].numpy()[::2]
    um_receivers = edges[1].numpy()[::2]

    #outer loop ensures multiple vertices can be created
    while um_labels.size != 0:
        max_edge = np.argmax(um_labels)

        #if none of the remaining edges pass the threshold, the jet has no more secondary vertices
        if um_labels[max_edge] < score_threshold:
            break

        vertex = np.array([um_senders[max_edge], um_receivers[max_edge]])

        #remove tracks and edges that are already matched to a vertex
        um_tracks = um_tracks[np.logical_not(np.isin(um_tracks, vertex))]
        um_labels = np.delete(um_labels, max_edge)
        um_senders = np.delete(um_senders, max_edge)
        um_receivers = np.delete(um_receivers, max_edge)

        #inner loop adds tracks to current vertex
        while um_labels.size != 0:

            #check average score of edges between each unmatched node and nodes already in the given vertex -> choose maximum and check against threshold
            max_score = [0,0]
            for i in um_tracks:
                relevant_edges = np.where(np.logical_or(np.logical_and((um_senders == i), np.isin(um_receivers, vertex)),np.logical_and((um_receivers == i), np.isin(um_senders, vertex))))[0]
                av_score = np.sum(um_labels[relevant_edges])/len(relevant_edges)
                if av_score > max_score[0]: max_score = [av_score, i]

            #if none of the unmatched vertices have average scores that pass the threshold, there are no more tracks associated with this vertex
            if max_score[0] < score_threshold:
                break

            #otherwise, add node with maximum score to vertex and remove all internal SV edges from the arrays
            vertex = np.append(vertex, max_score[1])
            um_tracks = um_tracks[np.logical_not(np.isin(um_tracks, vertex))]
            delete_indices = np.where(np.logical_and(np.isin(um_senders, vertex), np.isin(um_receivers, vertex)))[0]
            um_labels = np.delete(um_labels, delete_indices)
            um_senders = np.delete(um_senders, delete_indices)
            um_receivers = np.delete(um_receivers, delete_indices)

        #remove the rest of the edges associated with nodes already part of a vertex
        delete_indices = np.where(np.logical_or(np.isin(um_senders, vertex), np.isin(um_receivers, vertex)))[0]
        um_labels = np.delete(um_labels, delete_indices)
        um_senders = np.delete(um_senders, delete_indices)
        um_receivers = np.delete(um_receivers, delete_indices)

        vertex = np.sort(vertex)
        if vertex.size != 0: vertices.append(vertex)

    return vertices


#compare reco vertices to true and get metrics
def compare_vertices(true_vertices, reco_vertices):

    vertex_cm = np.zeros((3,3), dtype=int) #no sv, one sv, more than one sv (first index = true, second index = predicted)
    vertex_metrics = []
    vertex_assoc = np.empty(len(true_vertices), dtype=int)
    vertex_assoc.fill(-1) #true vertices with -1 have no reco association

    #count how many jets were correctly identified as containing an SV
    if len(true_vertices) == 0:
        if len(reco_vertices) == 0: vertex_cm[0,0] = 1
        elif len(reco_vertices) == 1: vertex_cm[0,1] = 1
        else: vertex_cm[0,2] = 1
    elif len(true_vertices) == 1:
        if len(reco_vertices) == 0: vertex_cm[1,0] = 1
        elif len(reco_vertices) == 1: vertex_cm[1,1] = 1
        else: vertex_cm[1,2] = 1
    else:
        if len(reco_vertices) == 0: vertex_cm[2,0] = 1
        elif len(reco_vertices) == 1: vertex_cm[2,1] = 1
        else: vertex_cm[2,2] = 1

    #associate reco vertices with truth vertices based on which reco and truth vertices have the most tracks in common
    assoc_true = [] #store true vertices that have already been associated
    assoc_reco = [] #store reco vertices that have already been associated

    for i in range(min(len(true_vertices), len(reco_vertices))):
        maximum_pair = [-1,-1]
        maximum_value = -1
        for j in range(len(true_vertices)):
            for k in range(len(reco_vertices)):
                matching_tracks = np.intersect1d(reco_vertices[k], true_vertices[j]).size
                if matching_tracks > maximum_value and j not in assoc_true and k not in assoc_reco:
                    maximum_pair = [j,k]
                    maximum_value = matching_tracks
        vertex_assoc[maximum_pair[0]] = maximum_pair[1]
        assoc_true.append(j)
        assoc_reco.append(k)

    #check what percentage of tracks were associated correctly in correctly identified SV's
    for i in range(len(true_vertices)):
        if vertex_assoc[i] != -1 and len(reco_vertices) > 0:
            reco_index = int(vertex_assoc[i])
            correct_tracks = np.intersect1d(reco_vertices[reco_index], true_vertices[i])
            vertex_metrics.append([correct_tracks.size/true_vertices[i].size, (reco_vertices[reco_index].size-correct_tracks.size)/reco_vertices[reco_index].size])

    return vertex_cm, vertex_metrics, vertex_assoc


def print_output(multi_class, cm):
    if not multi_class:
        print('\nTesting results:')
        print('             ||  Pred False  |  Pred True   |')
        print('---------------------------------------------')
        print(f'Actual False || {cm[0,0]:12} | {cm[0,1]:12} |')
        print(f'Actual True  || {cm[1,0]:12} | {cm[1,1]:12} |')
        print('---------------------------------------------')
        print('Accuracy: {:.4f}'.format((cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[0,1]+cm[1,0]))) #(tp+tn)/(tp+tn+fp+fn)
        print('Fake Rate (1-Precision): {:.4f}'.format(1.-cm[1,1]/(cm[1,1]+cm[0,1]))) #1-tp/(tp+fp)
        print('Efficiency (TPR): {:.4f}'.format(cm[1,1]/(cm[1,1]+cm[1,0]))) #tp/(tp+fn)
        print('True Negative Rate: {:.4f}'.format(cm[0,0]/(cm[0,0]+cm[0,1]))) #tn/(tn+fp)
        print('F1 Score {:.4f}\n'.format(2*cm[1,1]/(2*cm[1,1]+cm[0,1]+cm[1,0]))) #2*tp/(2*tp+fp+fn)
    else:
        print('\nTesting results:')
        print('       ||    Pred 0    |    Pred 1    |    Pred 2    |    Pred 3    |    Pred 4    ||  Recall ')
        print('----------------------------------------------------------------------------------------------')
        print(f'True 0 || {cm[0,0]:12d} | {cm[0,1]:12d} | {cm[0,2]:12d} | {cm[0,3]:12d} | {cm[0,4]:12d} || {cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]):.4f}')
        print(f'True 1 || {cm[1,0]:12d} | {cm[1,1]:12d} | {cm[1,2]:12d} | {cm[1,3]:12d} | {cm[1,4]:12d} || {cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]):.4f}')
        print(f'True 2 || {cm[2,0]:12d} | {cm[2,1]:12d} | {cm[2,2]:12d} | {cm[2,3]:12d} | {cm[2,4]:12d} || {cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]):.4f}')
        print(f'True 3 || {cm[3,0]:12d} | {cm[3,1]:12d} | {cm[3,2]:12d} | {cm[3,3]:12d} | {cm[3,4]:12d} || {cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]):.4f}')
        print(f'True 4 || {cm[4,0]:12d} | {cm[4,1]:12d} | {cm[4,2]:12d} | {cm[4,3]:12d} | {cm[4,4]:12d} || {cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]):.4f}')
        print('----------------------------------------------------------------------------------------------')
        print(f'Prec   ||       {cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]+cm[4,0]):.4f} |       {cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]+cm[4,1]):.4f} |       {cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]+cm[4,2]):.4f} |       {cm[3,3]/(cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]+cm[4,3]):.4f} |       {cm[4,4]/(cm[0,4]+cm[1,4]+cm[2,4]+cm[3,4]+cm[4,4]):.4f} ||\n')


def plot_metric_hist(hist_list, outfile_name, ext):
    canv1 = TCanvas("c1", "c1", 800, 600)
    canv1.SetGrid()
    colorlist = [1,4,8,2,6]
    gStyle.SetOptStat(0)
    legend = TLegend(0.78,0.95-0.1*max(len(hist_list),2),0.98,0.95)

    for i in range(len(hist_list)):
        legend.AddEntry(hist_list[i], "#splitline{%s}{#splitline{%d entries}{mean=%.2f}}"%(hist_list[i].GetName(), hist_list[i].GetEntries(), hist_list[i].GetMean()), "l")
        hist_list[i].SetLineColorAlpha(colorlist[i],0.65)
        hist_list[i].SetLineWidth(3)
        if hist_list[i].GetEntries(): hist_list[i].Scale(1./(hist_list[i].GetEntries()))
        hist_list[i].SetMaximum(1.2)
        if i == 0: hist_list[i].Draw()
        else:hist_list[i].Draw("SAMES")
    legend.SetTextSize(0.025)
    legend.Draw("SAME")
    canv1.SaveAs(outfile_name+ext)
    canv1.Clear()


def plot_profile(profile_list, outfile_name, labels, ylimit, ext):
    canv1 = TCanvas("c1", "c1", 800, 600)
    canv1.SetGrid()
    colorlist = [1,4,8,2,6]
    gStyle.SetOptStat(0)
    legend = TLegend(0.78,0.95-0.1*max(len(profile_list),2),0.98,0.95)

    for i in range(len(profile_list)):
        legend.AddEntry(profile_list[i], "#splitline{%s}{#splitline{%d entries}{y-mean=%.2f}}"%(labels[i], profile_list[i].GetEntries(), profile_list[i].GetMean(2)), "l")
        profile_list[i].SetLineColorAlpha(colorlist[i],0.65)
        profile_list[i].SetLineWidth(3)
        profile_list[i].SetMaximum(ylimit[1])
        profile_list[i].SetMinimum(ylimit[0])
        if i == 0: profile_list[i].Draw()
        else:profile_list[i].Draw("SAMES")
    legend.SetTextSize(0.025)
    legend.Draw("SAME")
    canv1.SaveAs(outfile_name+ext)
    canv1.Clear()
