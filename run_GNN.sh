#!/bin/bash

RUN=25
DATADIR=/global/cfs/cdirs/atlas/jmw464/data/
OUTPUTDIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/
DATA=btag_05_19_cut_v4.1c

EPOCHS=50

NORMED=1
MULTICLASS=0

#create output directory if not already there
if [ ! -d ${OUTPUTDIR}${RUN}/ ]
then
	mkdir ${OUTPUTDIR}${RUN}/
fi

source activate dgl-env

printf "##########BEGINNING TRAINING##########\n"
python scripts/GNN_main.py -r $RUN -e $EPOCHS -s $DATA -d ${DATADIR}${DATA}/ -o $OUTPUTDIR -n $NORMED -m $MULTICLASS

if [[ $MULTICLASS == 0 ]]
then
	python scripts/compare_performance.py -r $RUN -s $DATA -d ${DATADIR}${DATA}/ -o $OUTPUTDIR -n $NORMED
fi
