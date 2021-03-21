#!/usr/bin/env python

#GNN for secondary vertex reconstructions

from GNN_model import *

from torch_geometric.data import Data, DataLoader
import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt
import time

#th.set_printoptions(edgeitems=10000)
np.set_printoptions(threshold=sys.maxsize)

max_entries = 10
ngfeatures = 0 #number of features for graph
nnfeatures = 13 #number of features per node
nefeatures = 1 #number of features per edge
nepochs = 5

trainp = .7
valp = .2
testp = .1
learning_rate = 0.01
pos_weight = th.tensor([25]) #reweight positive labels since negatives outnumber positives by 25:1
reco_mode = 'b' #pv, sv or b (reconstruct only primary vertices, only secondary vertices or both)


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


#NO LONGER USED
def create_mask(n, train_split, test_split):
    train_mask = np.zeros(n, dtype=np.bool)
    val_mask = np.zeros(n, dtype=np.bool)
    test_mask = np.zeros(n, dtype=np.bool)
    
    train_mask[:round(n*train_split)] = 1
    np.random.shuffle(train_mask)
    taken_indices = np.array(np.where(train_mask > 0))[0]
    indices = np.array(np.where(1-train_mask > 0))[0]
    np.random.shuffle(indices)
    indices_test = (indices[:round(n*test_split)])
    indices_val = (indices[round(n*test_split):])

    for i in range(n):
        if i in indices_test:
            test_mask[i] = 1
        elif i in indices_val:
            val_mask[i] = 1

    return train_mask, val_mask, test_mask


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(argv):
    gROOT.SetBatch(True)
    
    start_time = time.time()
    infile = h5py.File("/global/homes/j/jmw464/ATLAS/cuts.hdf5", "r")
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
            node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q, jet_pt, jet_eta, jet_phi, track_vx, track_vy, track_vz]
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

        if ntracks > 1:
            #tr_mask, v_mask, te_mask = create_mask(nedges, trainp, testp)
            e_index = th.from_numpy(np.array(create_edge_list(ntracks)))
            g = Data(x=th.from_numpy(node_features), edge_index=e_index, edge_attr=th.from_numpy(edge_features), y=th.from_numpy(truth_labels))#, train_mask=th.from_numpy(tr_mask), val_mask=th.from_numpy(v_mask), test_mask=th.from_numpy(te_mask)) 
            g_list.append(g)

    print("Time: {}s".format(time.time() - start_time))
    print("TRUTH {}".format(total_ones/total_edges))

    train_list = g_list[:int(len(g_list)*trainp-1)]
    val_list = g_list[int(len(g_list)*trainp-1):int(len(g_list)*(trainp+valp)-1)]
    test_list = g_list[int(len(g_list)*(trainp+valp)-1):]

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = EdgePredModel(nnfeatures, 500, 100).to(device)
    
    opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    train_loss_array = np.zeros(nepochs)
    val_loss_array = np.zeros(nepochs)
    sig = nn.Sigmoid()

    print("Running on {}".format(device))

    train_loader = DataLoader(train_list, batch_size=500, shuffle=True)
    val_loader = DataLoader(val_list, batch_size = 100, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=1, shuffle=False)

    for name, param in model.named_parameters():
        print(name, param.data, param.requires_grad)
    print("")

    model = model.double()
    for epoch in range(1,nepochs+1):
        print("Epoch: {}".format(epoch))
        
        #training loop
        model.train()
        for batch in train_loader:
            batch = batch.to(device) #transfer batch to relevant device
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index, device)
            l_pred = loss(pred.float(),batch.y.float())
            opt.zero_grad()
            l_pred.backward()
            opt.step()
            print(l_pred.item())
            train_loss_array[epoch-1] += l_pred.item()/len(train_loader)

        #validation loop
        model.eval()
        tp = tn = fp = fn = 0
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, device)
            l_pred = loss(pred.float(),batch.y.float())
            pred = sig(pred.float()).cpu().detach().numpy().flatten().round().astype(int)
            true = batch.y.cpu().numpy().flatten().astype(int)
            tp += np.sum((true == 1) & (pred == 1))
            tn += np.sum((true == 0) & (pred == 0))
            fp += np.sum((true == 0) & (pred == 1))
            fn += np.sum((true == 1) & (pred == 0))
            val_loss_array[epoch-1] += l_pred.item()/len(val_loader)
        
        print('Predicted: {} true, {} false; Truth: {} true, {} false'.format(tp+fp, tn+fn, tp+fn, tn+fp))
        print('Accuracy: {:.4f}'.format((tp+tn)/(tp+tn+fp+fn)))
        print('Fake Rate (1-Precision): {:.4f}'.format(1.-tp/(tp+fp)))
        print("Time: {}s".format(time.time() - start_time))
        print("")

    #testing loop
    tp = tn = fp = fn = 0
    model.eval()
    for batch in test_loader:
        batch = batch.to(device)
        pred = sig(model(batch.x, batch.edge_index, device).float()).cpu().detach().numpy().flatten().round().astype(int)
        true = batch.y.cpu().numpy().flatten().astype(int)
        tp += np.sum((true == 1) & (pred == 1))
        tn += np.sum((true == 0) & (pred == 0))
        fp += np.sum((true == 0) & (pred == 1))
        fn += np.sum((true == 1) & (pred == 0))

    acc = (tp+tn)/(tp+tn+fp+fn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    prec = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    pos_tot = tp+fn
    neg_tot = tn+fp
    pos_pred = tp+fp
    neg_pred = tn+fn
    print('TEST')
    print('Predicted: {} true, {} false; Truth: {} true, {} false'.format(pos_pred, neg_pred, pos_tot, neg_tot))
    print('Accuracy: {:.4f}'.format(acc))
    print('Fake Rate (1-Precision): {:.4f}'.format(1.-prec))
    print('Efficiency (TPR): {:.4f}'.format(tpr))
    print('True Negative Rate: {:.4f}'.format(tnr))
    print('F1 Score {:.4f}'.format(f1))

    plt.ioff()
    plt.plot(range(nepochs), train_loss_array, label="Training")
    plt.plot(range(nepochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("lossplot.png")


if __name__ == '__main__':
    main(sys.argv)
