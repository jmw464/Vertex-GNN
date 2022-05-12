#!/usr/bin/env python

######################################### GNN_eval.py #########################################
# PURPOSE: contains helper functions related to GNN evaluation
# EDIT TO: change definition of "badly reconstructed" jet
# ------------------------------------------Summary--------------------------------------------
# This script contains a collection of functions used throughout the evaluation of GNN
# performance and comparisons to SV1 (used in GNN_main, plot_results and compare_performance).
###############################################################################################


import numpy as np
import torch as th
import dgl


#evaluate confusion matrix for binary case
def evaluate_confusion_bin(true, pred):

    cm = np.zeros((2,2),dtype=int)
    cm[1,1] = np.sum((true == 1) & (pred == 1)) #true positive - actually true, marked as true
    cm[0,0] = np.sum((true == 0) & (pred == 0)) #true negative - actually false, marked as false
    cm[0,1] = np.sum((true == 0) & (pred == 1)) #false positive - actually false, marked as true
    cm[1,0] = np.sum((true == 1) & (pred == 0)) #false negative - actually true, marked as false
    
    return cm


#evaluate confusion matrix for multi-class case
def evaluate_confusion_mult(true, pred):

    cm = np.zeros((3,3),dtype=int)
    for i in range(3):
        for j in range(3):
            cm[i,j] = np.sum((true == i) & (pred == j))
    
    return cm


#check if GNN performed poorly on event
def is_bad_jet(graph, hist_list, multi_class, bin_threshold, mult_threshold):

    #store recall for each class
    if not multi_class:
        r_array = np.zeros(2)
        pred = th.round(graph.edata['pred']).cpu().detach().numpy().astype(int)
        true = graph.edata['bin_labels'].cpu().detach().numpy().astype(int)
        cm = evaluate_confusion_bin(true, pred)
        r_threshold = bin_threshold
    else:
        r_array = np.zeros(5)
        pred = np.argmax(graph.edata['pred'].cpu().detach().numpy(), axis=1).astype(int)
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
            return True

    return False


#group tracks into vertices based on edge scores - truth mode generates truth SVs based on training target labels
def find_vertices_bin(graph, mode, score_threshold):
    ntracks = graph.number_of_nodes()
    edges = graph.all_edges()
    um_tracks = np.array(range(ntracks))
    um_senders = np.zeros(int(ntracks*(ntracks-1)/2), dtype=int)
    um_receivers = np.zeros(int(ntracks*(ntracks-1)/2), dtype=int)
    um_labels = np.zeros(int(ntracks*(ntracks-1)/2))

    if mode == 'truth' or mode == 't':
        labels = graph.edata['bin_labels'].cpu().detach().numpy().flatten()
    else:
        labels = graph.edata['pred'].cpu().detach().numpy().flatten()

    vertices = []

    #average out forward and backward edges
    count_index = 0
    for i in range(ntracks):
        for j in range(ntracks):
            if i < j:
                um_senders[count_index] = i
                um_receivers[count_index] = j
                edge_index_f = np.argwhere(np.logical_and(edges[0].numpy() == i, edges[1].numpy() == j))
                edge_index_b = np.argwhere(np.logical_and(edges[0].numpy() == j, edges[1].numpy() == i))
                um_labels[count_index] = (labels[edge_index_b]+labels[edge_index_f])/2
                count_index += 1

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
    vertex_metrics = np.zeros((len(true_vertices),len(reco_vertices),3)) #array containing number of common tracks, efficiency and fake rate for each true/reco vertex combo
    vertex_metrics[:,:,2].fill(1) #set initial fake rate to 1

    #count how many jets were correctly identified as containing an SV
    true_index = min(2,len(true_vertices))
    reco_index = min(2,len(reco_vertices))
    vertex_cm[true_index, reco_index] = 1

    #calculate efficiency, fake rate for each vertex combo
    for i in range(len(true_vertices)):
        for j in range(len(reco_vertices)):
            matching_tracks = np.intersect1d(true_vertices[i], reco_vertices[j]).size
            vertex_eff = matching_tracks/true_vertices[i].size
            vertex_fr = (reco_vertices[j].size-matching_tracks)/reco_vertices[j].size
            vertex_metrics[i,j] = [matching_tracks, vertex_eff, vertex_fr]
    
    return vertex_cm, vertex_metrics


#turn vertex metrics into direct associations between truth and reco vertices - mode = 't' (associate each truth to a reco vertex) or 'r' (associate each reco vertex to a truth vertex)
def associate_vertices(vertex_metrics, mode):
    if mode == 'r':
        vertex_assoc_array = np.zeros(vertex_metrics.shape[1],dtype=int)
        vertex_assoc_array.fill(-1)
        if vertex_metrics.size > 0:
            for i in range(vertex_metrics.shape[1]):
                vertex_assoc = np.argwhere(np.logical_and(vertex_metrics[:,i,0] == np.amax(vertex_metrics[:,i,0]), vertex_metrics[:,i,0] > 0)).flatten()
                if vertex_assoc.size > 1: vertex_assoc = np.argmax(vertex_metrics[vertex_assoc,i,1])
                if vertex_assoc.size > 1: vertex_assoc = np.argmin(vertex_metrics[vertex_assoc,i,2])
                if vertex_assoc.size > 1: vertex_assoc = vertex_assoc[0]
                if vertex_assoc.size > 0: vertex_assoc_array[i] = vertex_assoc
    elif mode == 't':
        vertex_assoc_array = np.zeros(vertex_metrics.shape[0],dtype=int)
        vertex_assoc_array.fill(-1)
        if vertex_metrics.size > 0:
            for i in range(vertex_metrics.shape[0]):
                vertex_assoc = np.argwhere(np.logical_and(vertex_metrics[i,:,0] == np.amax(vertex_metrics[i,:,0]), vertex_metrics[i,:,0] > 0)).flatten()
                if vertex_assoc.size > 1: vertex_assoc = np.argmax(vertex_metrics[i,vertex_assoc,1])
                if vertex_assoc.size > 1: vertex_assoc = np.argmin(vertex_metrics[i,vertex_assoc,2])
                if vertex_assoc.size > 1: vertex_assoc = vertex_assoc[0]
                if vertex_assoc.size > 0: vertex_assoc_array[i] = vertex_assoc
    else:
        print("INVALID MODE SELECTION")

    return vertex_assoc_array


#print confusion matrix and other GNN evaluation metrics
def print_output(multi_class, cm, label):
    if not multi_class:
        print(label+' results:')
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
        print(label+' results:')
        print('       ||    Pred 0    |    Pred 1    |    Pred 2    ||  Recall ')
        print('----------------------------------------------------------------------------------------------')
        print(f'True 0 || {cm[0,0]:12d} | {cm[0,1]:12d} | {cm[0,2]:12d} || {cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]):.4f}')
        print(f'True 1 || {cm[1,0]:12d} | {cm[1,1]:12d} | {cm[1,2]:12d} || {cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]):.4f}')
        print(f'True 2 || {cm[2,0]:12d} | {cm[2,1]:12d} | {cm[2,2]:12d} || {cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]):.4f}')
        print('----------------------------------------------------------------------------------------------')
        print(f'Prec   ||       {cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]):.4f} |       {cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]):.4f} |       {cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]):.4f} ||\n')


def print_weight_contribution(model, feature_list):
    separate_srcdst = True
    gnnweights = mlpweights = None
    for name, param in model.named_parameters():
        if name == "gcn.conv1.fc.weight":
            gnnweights = th.sum(th.abs(model.gcn.conv1.fc.weight), 0)/model.gcn.conv1.fc.weight.shape[0]
        elif name == "gcn.conv1.weight":
            gnnweights = th.sum(th.abs(model.gcn.conv1.weight), 0)/model.gcn.conv1.weight.shape[0]
        elif name == "gcn.conv1.fc_src.weight":
            gnnweights = th.sum((th.abs(model.gcn.conv1.fc_src.weight) + th.abs(model.gcn.conv1.fc_dst.weight)), 0)/(2*model.gcn.conv1.fc_src.weight.shape[0])
        elif name == "nodemlp.lin.0.weight":
            mlpweights = th.sum(th.abs(model.nodemlp.lin[0].weight), 0)/model.nodemlp.lin[0].weight.shape[0]
    
    print("Printing average of absolute value of weights associated with each feature in first network layer:")
    if gnnweights != None and mlpweights != None:
        for i,feature in enumerate(feature_list):
            print(feature+": {} (NodeMLP), {} (GraphNN)".format(mlpweights[i], gnnweights[i]))
    elif gnnweights != None:
        for i,feature in enumerate(feature_list):
            print(feature+": {} (GraphNN)".format(gnnweights[i]))
    elif mlpweights != None:
        for i,feature in enumerate(feature_list):
            print(feature+": {} (NodeMLP)".format(mlpweights[i]))
