#!/usr/bin/env python

#GNN for secondary vertex reconstructions

from GNN_model import *

import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt
import time

#th.set_printoptions(edgeitems=10000)
#np.set_printoptions(threshold=sys.maxsize)

#input data parameters
infile_name = "/global/homes/j/jmw464/ATLAS/test.hdf5"
max_entries = 1000000
nnfeatures = 10 #number of features per node
nefeatures = 1 #number of features per edge -- NOT CURRENTLY USED

#training parameters
nepochs = 50
valp = .2 #fraction of data reserved for validation
testp = .1 #fraction of data reserved for testing
learning_rate = 0.001
batch_size = 500 #number of jets in a single training batch

#model parameters
attention_heads = 2 #number of attention heads in GAT layer -> these are averaged over
gnn_hidfeats = 256 #number of hidden features in GAT layer output
mlp_hidfeats = 512 #number of hidden features in MLP layers (actual number is twice this since two node feature sets are concatenated)

#script options
reweight = True #reweight positive labels in loss to make positives and negatives equally important
reco_mode = 'b' #pv, sv or b (reconstruct only primary vertices, only secondary vertices or both)
load_checkpoint = False
checkpoint_path = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/model.pt"

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


#evaluate tp, tn, fp, fn for GNN results
def evaluate_results(true, pred):

    tp = np.sum((true == 1) & (pred == 1))
    tn = np.sum((true == 0) & (pred == 0))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))
    
    return tp, tn, fp, fn


def main(argv):
    gROOT.SetBatch(True)
    
    print("")
    print("Processing input data.")

    start_time = time.time()
    infile = h5py.File(infile_name, "r")
    g_list = []

    ievent = -1
    total_jets = len(infile['info']['event'])
    current_event = -1
    track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
    total_ones = 0
    total_edges = 0
    for ientry in range(total_jets):
        
        #read in current event number and update number of processed events - events are not necessarily sequential
        if current_event != infile['info']['event'][ientry]:
            current_event = infile['info']['event'][ientry]
            ievent += 1

        if ievent >= max_entries:
            break
        
        #read in event/jet information
        current_jet = infile['info']['jet'][ientry]
        ntracks =  infile['info']['ntracks'][ientry]
        primary_vertex = np.array([infile['efeatures']['event_vx'][current_event], infile['efeatures']['event_vy'][current_event], infile['efeatures']['event_vz'][current_event]])
        nedges = ntracks*(ntracks-1)

        #initialize empty arrays
        node_features = np.zeros((ntracks,nnfeatures))
        edge_features = np.zeros((nedges,nefeatures))
        vertex_positions = np.zeros((ntracks,3))
        truth_labels = np.zeros((nedges,1))
        #print("event %d, jet %d with %d tracks"%(ievent, current_jet, ntracks))
        
        #read in features
        for j in range(ntracks):
            track_pt  = infile['tfeatures']['pt'][track_offset+j]/1000 #convert to GeV
            track_eta = infile['tfeatures']['eta'][track_offset+j]
            track_theta = infile['tfeatures']['theta'][track_offset+j]
            track_phi = infile['tfeatures']['phi'][track_offset+j]
            track_d0 = infile['tfeatures']['d0'][track_offset+j]
            track_z0 = infile['tfeatures']['z0'][track_offset+j]
            track_q = infile['tfeatures']['q'][track_offset+j]

            jet_pt = infile['jfeatures']['pt'][ientry]/1000 #convert to GeV
            jet_eta = infile['jfeatures']['eta'][ientry]
            jet_phi = infile['jfeatures']['phi'][ientry]

            track_vx = infile['labels']['track_vx'][track_offset+j]
            track_vy = infile['labels']['track_vy'][track_offset+j]
            track_vz = infile['labels']['track_vz'][track_offset+j]
            node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q, jet_pt, jet_eta, jet_phi]#, track_vx, track_vy, track_vz]
            vertex_positions[j] = [track_vx, track_vy, track_vz]
        
        track_offset += ntracks

        #calculate edge features and truth labels
        counter = 0
        for j in range(ntracks):

            #set PV condition on truth labels
            pv_distance = np.linalg.norm(vertex_positions[j]-primary_vertex)
            if reco_mode == 'pv':
                pv_condition = (pv_distance < 1e-4)
            elif reco_mode == 'sv':
                pv_condition = (pv_distance > 1e-4)
            else:
                pv_condition = True

            for k in range(j+1, ntracks):
                
                #edge features
                delta_pt = abs(node_features[j][0] - node_features[k][0])
                edge_features[counter:counter+2] = [delta_pt]
                   
                #truth labels - vertices have to be close, real (not -999) and fulfill PV condition
                distance = np.linalg.norm(vertex_positions[j]-vertex_positions[k])
                if (distance < 1e-4) and pv_condition and vertex_positions[j][0] > -900.:
                    truth_labels[counter:counter+2] = 1
                else:
                    truth_labels[counter:counter+2] = 0
                    
                counter += 2
       
        total_ones += np.sum(truth_labels)
        total_edges += np.size(truth_labels)

        #create graph objects and append them to the list
        if ntracks > 1:
            g = dgl.graph((create_edge_list(ntracks)))
            g.ndata['features'] = th.from_numpy(node_features)
            g.edata['labels'] = th.from_numpy(truth_labels)
            g_list.append(g)

    infile.close()
    p_time = time.time()-start_time
    print("Finished processing input data. Time elapsed: {}s.\n".format(p_time))

    #reweight positive labels automatically if desired
    if reweight:
        truth_frac = total_ones/total_edges
        pos_weight = th.tensor([(1-truth_frac)/truth_frac])
    else:
        pos_weight = th.tensor([1])

    #split data into testing, training and validation set
    test_list = g_list[:int(len(g_list)*testp-1)]
    val_list = g_list[int(len(g_list)*testp-1):int(len(g_list)*(testp+valp)-1)]
    train_list = g_list[int(len(g_list)*(testp+valp)-1):]

    device = th.device('cuda' if th.cuda.is_available() else 'cpu') #automatically run on GPU if available
    
    model = EdgePredModel(nnfeatures, gnn_hidfeats, mlp_hidfeats, attention_heads).double().to(device)
    opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    sig = nn.Sigmoid() #since loss already has sigmoid function built in, we need to pass model outputs through a separate sigmoid function for evaluation

    train_loss_array = np.zeros(nepochs)
    val_loss_array = np.zeros(nepochs)

    #split testing data into batches
    train_batch = []
    b_index = 0
    for i in range(math.floor(len(train_list)/batch_size)):
        train_batch.append(dgl.batch(train_list[batch_size*i:batch_size*(i+1)]))
        b_index = i
    train_batch.append(dgl.batch(train_list[batch_size*b_index:]))

    val_batch = dgl.batch(val_list)
    test_batch = dgl.batch(test_list)
    
    #print model parameters
    print("Model built. Parameters:")
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)
    print("")

    #load existing checkpoint
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("Loading previous model. Starting from epoch {}.".format(start_epoch))
    else:
        start_epoch = 1

    #main training loop
    t_time = time.time()-start_time
    print("Beginning training. Running on {}. Time elapsed: {}s.\n".format(device, t_time))
    for epoch in range(start_epoch,nepochs+1):
        print("Epoch: {}".format(epoch))
        
        #training
        model.train()
        for batch in train_batch:
            batch = batch.to(device) #transfer batch to relevant device
            pred = model(batch, batch.ndata['features'])
            pred_l = loss(pred, batch.edata['labels'])
            opt.zero_grad()
            pred_l.backward()
            opt.step()
            print("Training loss: {}".format(pred_l.item()))
            train_loss_array[epoch-1] += pred_l.item()/len(train_batch)

        #save checkpoint
        th.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, checkpoint_path)

        #validation
        model.eval()
        val_batch = val_batch.to(device)
        pred = model(val_batch, val_batch.ndata['features'])
        pred_l = loss(pred, val_batch.edata['labels'])
        print("Validation loss: {}".format(pred_l.item()))
        val_loss_array[epoch-1] = pred_l.item()

        pred = sig(pred.float()).cpu().detach().numpy().flatten().round().astype(int)
        true = val_batch.edata['labels'].cpu().numpy().flatten().astype(int)
        tp, tn, fp, fn = evaluate_results(true, pred)
        
        e_time = time.time()-start_time
        print('Validation results: {} TP, {} FP, {} TN, {} FN. Time elapsed: {}s.\n'.format(tp, fp, tn, fn, e_time))

    print("Training finished. Evaluating model.\n")

    #testing
    model.eval()
    test_batch = test_batch.to(device)
    node_features = test_batch.ndata['features']
    edge_labels = test_batch.edata['labels']
    
    pred = sig(model(test_batch, test_batch.ndata['features']).float()).cpu().detach().numpy().flatten().round().astype(int)
    true = test_batch.edata['labels'].cpu().numpy().flatten().astype(int)
    tp, tn, fp, fn = evaluate_results(true, pred)
    print("Test results: {} TP, {} FP, {} TN, {} FN.".format(tp, fp, tn, fn))
    print('Accuracy: {:.4f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Fake Rate (1-Precision): {:.4f}'.format(1.-tp/(tp+fp)))
    print('Efficiency (TPR): {:.4f}'.format(tp/(tp+fn)))
    print('True Negative Rate: {:.4f}'.format(tn/(tn+fp)))
    print('F1 Score {:.4f}'.format(2*tp/(2*tp+fp+fn)))

    #plot loss
    plt.ioff()
    plt.plot(range(nepochs), train_loss_array, label="Training")
    plt.plot(range(nepochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("lossplot.png")


if __name__ == '__main__':
    main(sys.argv)
