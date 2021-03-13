#!/usr/bin/env python

#GNN for secondary vertex reconstructions

from GNN_model import *

from torch_geometric.data import Data, DataLoader
import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

max_entries = 1000
ngfeatures = 0 #number of features for graph
nnfeatures = 10 #number of features per node
nefeatures = 1 #number of features per edge
nepochs = 20

trainp = .8
testp = .2
learning_rate = 0.01

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
    
    infile = h5py.File("/global/homes/j/jmw464/ATLAS/cuts.hdf5", "r")
    g_list = []

    ievent = -1
    total_jets = len(infile['info']['event'])
    current_event = -1
    track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
    total_ones = 0
    total_edges = 0
    for ientry in range(total_jets):
        
        if current_event != infile['info']['event'][ientry]:
            current_event = infile['info']['event'][ientry]
            ievent += 1

        if ievent >= max_entries:
            break

        current_jet = infile['info']['jet'][ientry]
        ntracks =  infile['info']['ntracks'][ientry]
        nedges = ntracks*(ntracks-1)
        node_features = np.zeros((ntracks,nnfeatures))
        edge_features = np.zeros((nedges,nefeatures))
        vertex_positions = np.zeros((ntracks,3))
        truth_labels = np.zeros((nedges,1))
        print("event %d, jet %d with %d tracks"%(ievent, current_jet, ntracks))
        
        #read in features
        for j in range(ntracks):
            track_pt  = infile['tfeatures']['pt'][track_offset+j]
            track_eta = infile['tfeatures']['eta'][track_offset+j]
            track_theta = infile['tfeatures']['theta'][track_offset+j]
            track_phi = infile['tfeatures']['phi'][track_offset+j]
            track_d0 = infile['tfeatures']['d0'][track_offset+j]
            track_z0 = infile['tfeatures']['z0'][track_offset+j]
            track_q = infile['tfeatures']['q'][track_offset+j]

            jet_pt = infile['jfeatures']['pt'][ientry]
            jet_eta = infile['jfeatures']['eta'][ientry]
            jet_phi = infile['jfeatures']['phi'][ientry]

            track_vx = infile['labels']['track_vx'][track_offset+j]
            track_vy = infile['labels']['track_vy'][track_offset+j]
            track_vz = infile['labels']['track_vz'][track_offset+j]
            node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q, jet_pt, jet_eta, jet_phi]#, track_vx, track_vy, track_vz]
            vertex_positions[j] = [track_vx, track_vy, track_vz]
        track_offset += ntracks

        #calculate edge features
        counter = 0
        for j in range(ntracks):
            for k in range(j+1, ntracks):
                delta_pt = abs(node_features[j][0] - node_features[k][0])
                edge_features[counter:counter+2] = [delta_pt]
                    
                distance = np.linalg.norm(vertex_positions[j]-vertex_positions[k])
                if distance < 1e-4:
                    truth_labels[counter:counter+2] = 1
                else:
                    truth_labels[counter:counter+2] = 0
                    
                counter += 2

        total_ones += np.sum(truth_labels)
        total_edges += np.size(truth_labels)

        if ntracks > 1:
            tr_mask, v_mask, te_mask = create_mask(nedges, trainp, testp)
            e_index = th.from_numpy(np.array(create_edge_list(ntracks)))
            g = Data(x=th.from_numpy(node_features), edge_index=e_index, edge_attr=th.from_numpy(edge_features), y=th.from_numpy(truth_labels))#, train_mask=th.from_numpy(tr_mask), val_mask=th.from_numpy(v_mask), test_mask=th.from_numpy(te_mask)) 
            g_list.append(g)

    print("TRUTH {}".format(total_ones/total_edges))

    train_list = g_list[:int(len(g_list)*trainp-1)]
    test_list = g_list[int(len(g_list)*trainp-1):]
    
    train_loader = DataLoader(train_list, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=1, shuffle=False)

    model = EdgePredModel(nnfeatures, 64, 32)
    opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.BCELoss()
    loss_array = np.zeros(nepochs)

    for name, param in model.named_parameters():
        print(name, param.data, param.requires_grad)

    model = model.double()
    model.train()
    for epoch in range(nepochs):
        print("Epoch: {}".format(epoch))
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = loss(out.float(),batch.y.float())
            out.backward()
            opt.step()
            loss_array[epoch] += out.item()/len(train_loader)
            print(out.item())

    correct = 0
    correct_ones = 0
    total = 0
    total_ones = 0
    model.eval()
    for batch in test_loader:
        pred = model(batch.x, batch.edge_index).detach().numpy().flatten().round().astype(int)
        true = batch.y.numpy().flatten().astype(int)
        correct += np.sum(true == pred)
        correct_ones += np.sum((true == 1) & (pred == 1))
        total += np.size(true)
        total_ones += np.sum(true == 1)
        break

    acc = correct/total
    tpr = correct_ones/total_ones
    print('Accuracy: {:.4f}'.format(acc))
    print('True Positive Rate: {:.4f}'.format(tpr))

    plt.ioff()
    plt.plot(range(nepochs), loss_array, label="Training")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("lossplot.png")

if __name__ == '__main__':
    main(sys.argv)
