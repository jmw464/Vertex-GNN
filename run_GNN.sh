#!/bin/bash

RUN=3
NTUPLE=/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/raw/user.jmwagner.24900045.Akt4EMPf_BTagging201903._000007.root
DATADIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/
OUTPUTDIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/
DATA=btag_07_19_cut_v2s

EPOCHS=30
LR=0.001
ENTRIES=10000

NORMED=1
BADJETS=1

PROCESS=false
TRAIN=true
PLOT=true

#create output directory if not already there
if [ ! -d $OUTPUTDIR$RUN/ ]
then
	mkdir $OUTPUTDIR$RUN/
else
	printf "WARNING: Directory already exists and contents might be overwritten. Continue? (y/n)\n"
	read YN
	printf "\n"
fi

if [ "$YN" == "N" ] || [ "$YN" == "n" ]
then
	exit 1
fi

source activate dgl-env

if $PROCESS
then
	printf "##########BEGINNING PROCESSING##########\n"
	printf "Running make_cuts.py to create $DATA.hdf5 with $ENTRIES jets\n"
	python scripts/make_cuts.py -n $NTUPLE -e $ENTRIES -d $DATADIR -s $DATA
	printf "\n"

	printf "Running create_graphs.py to transform $DATA.hdf5 into GNN compatible data\n"
	python scripts/create_graphs.py -d $DATADIR -s $DATA
	printf "\n"
	
	if [[ $NORMED != 0 ]]
	then
		printf "Running norm_graphs.py to normalize graphs based on training dataset\n"
		python scripts/norm_graphs.py -d $DATADIR -s $DATA
		printf "\n"
	fi
fi

if $TRAIN
then
	printf "##########BEGINNING TRAINING##########\n"
	python scripts/GNN_main.py -r $RUN -l $LR -e $EPOCHS -s $DATA -d $DATADIR -o $OUTPUTDIR -n $NORMED
fi

if $PLOT
then
	printf "##########BEGINNING PLOTTING##########\n"
	python scripts/generate_plots.py -r $RUN -e $ENTRIES -d $DATA -o $OUTPUTDIR -n $NTUPLE -b $BADJETS
fi
