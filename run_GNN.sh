#!/bin/bash

RUN=14
DATADIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/
OUTPUTDIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/
DATA=btag_05_19_cut_v3c_1

EPOCHS=30
LR=0.001

NORMED=1
BADJETS=1

#create output directory if not already there
if [ ! -d ${OUTPUTDIR}${RUN}/ ]
then
	mkdir ${OUTPUTDIR}${RUN}/
fi

source activate dgl-env

printf "##########BEGINNING TRAINING##########\n"
python scripts/GNN_main.py -r $RUN -l $LR -e $EPOCHS -s $DATA -d ${DATADIR}${DATA}/ -o $OUTPUTDIR -n $NORMED
