#!/usr/bin/env python

#GNN for secondary vertex reconstructions

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
import os,sys,math,glob,ROOT
import numpy as np
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

max_entries = 10000
ngfeatures = 0 #number of features for graph
nnfeatures = 7 #number of features per node
nefeatures = 1 #number of features per edge
nepochs = 20

trainp = .7
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


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_features, out_features)
        #self.conv2 = GCNConv(hidden_features, int(hidden_features/2))
        #self.conv3 = GCNConv(int(hidden_features/2), out_features)

    def forward(self, x, edge_index):
        # inputs are features of nodes
        h = self.conv1(x, edge_index)
        h = h.tanh()
        #h = self.conv2(h, edge_index)
        #h = h.tanh()
        #h = self.conv3(h, edge_index)
        #h = h.tanh()
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
    def forward(self, x, edge_index):
        nedges = list(edge_index.shape)[1]
        he = th.empty([nedges, self.in_features])
        for i in range(nedges):
            edge = edge_index[:,i]
            sender = edge[0]
            receiver = edge[1]
            he[i] = th.cat((x[sender],x[receiver]))

        he = he.double()
        he = self.lin1(he)
        he = he.tanh()
        he = self.lin2(he)
        he = he.tanh()

        return he


class DotProductPredictor(nn.Module):
    def forward(self, x, edge_index):
        nedges = list(edge_index.shape)[1]
        he = th.empty([nedges, 1])
        for i in range(nedges):
            edge = edge_index[:,i]
            sender = edge[0]
            receiver = edge[1]
            he[i] = th.dot(x[sender],x[receiver])
        return he


class EdgePredModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = GCN(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features*2, out_features, 1)#DotProductPredictor()
    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index)
        h = self.pred(h, edge_index)
        h = h.sigmoid()
        return h


def main(argv):
    gROOT.SetBatch(True)
    
    ntuple = TFile('/global/homes/t/toyamaza/workdir/ctag/data/ntuples/v9/output_ttbarAllHad_nominal.root')
    tree = ntuple.Get("bTag_AntiKt4EMPFlowJets_BTagging201903")
    
    g_list = []

    total_edges = 0
    for ientry,entry in enumerate(tree):
        njets = entry.njets

        for i in range(njets):
            ntracks =  entry.jet_trk_pt[i].size()
            nedges = ntracks*(ntracks-1)
            total_edges += nedges
            node_features = np.zeros((ntracks,nnfeatures))
            edge_features = np.zeros((nedges,nefeatures))
            vertex_positions = np.zeros((ntracks,3))
            truth_labels = np.zeros((nedges,1))
            print("event %d, jet %d with %d tracks"%(ientry, i, ntracks))
        
            #read in features
            for j in range(ntracks):
                track_pt  = entry.jet_trk_pt[i][j]
                track_eta = entry.jet_trk_eta[i][j]
                track_theta = entry.jet_trk_theta[i][j]
                track_phi = entry.jet_trk_phi[i][j]
                track_d0 = entry.jet_trk_d0[i][j]
                track_z0 = entry.jet_trk_z0[i][j]
                track_q = entry.jet_trk_charge[i][j]

                track_vx = entry.jet_trk_vtx_X[i][j]
                track_vy = entry.jet_trk_vtx_Y[i][j]
                track_vz = entry.jet_trk_vtx_Z[i][j]
                node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q]#, track_vx, track_vy, track_vz]
                vertex_positions[j] = [track_vx, track_vy, track_vz]

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

            if ntracks > 1:
                tr_mask, v_mask, te_mask = create_mask(nedges, trainp, testp)
                e_index = th.from_numpy(np.array(create_edge_list(ntracks)))
                g = Data(x=th.from_numpy(node_features), edge_index=e_index, edge_attr=th.from_numpy(edge_features), y=th.from_numpy(truth_labels), train_mask=th.from_numpy(tr_mask), val_mask=th.from_numpy(v_mask), test_mask=th.from_numpy(te_mask))
                g_list.append(g)

        if ientry >= max_entries-1:
            break

    print("TRUTH {}".format(np.sum(truth_labels)/np.size(truth_labels)))

    loader = DataLoader(g_list, batch_size=10, shuffle=True)

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
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = loss(out[batch.train_mask].float(),batch.y[batch.train_mask].float())
            out.backward()
            opt.step()
            loss_array[epoch] += out.item()/len(loader)
            print(out.item())

    correct = 0
    correct_ones = 0
    total = 0
    total_ones = 0
    model.eval()
    for batch in loader:
        pred = model(batch.x, batch.edge_index).detach().numpy().flatten().round().astype(int)
        true = batch.y.numpy().flatten().astype(int)
        mask = batch.test_mask.numpy()
        correct += np.sum(true[mask] == pred[mask])
        print(true)
        print(pred)
        print(true[mask])
        print(pred[mask])
        correct_ones += np.sum((true[mask] == 1) & (pred[mask] == 1))
        total += np.sum(mask)
        total_ones += np.sum(true[mask] == 1)
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
