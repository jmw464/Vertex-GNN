#!/bin/bash

NTUPLES="user.jmwagner.24900045.Akt4EMPf_BTagging201903._000007 user.jmwagner.24900045.Akt4EMPf_BTagging201903._000016"
DATADIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/
DATANAME=btag_07_19_cut_v2c

ENTRIES=100000 #jets used per file (after cuts)

NORMED=1

#create output directory if not already there
if [ ! -d ${DATADIR}${DATANAME}/ ]
then
	mkdir ${DATADIR}${DATANAME}/
else
	printf "WARNING: Data directory already exists and contents might be overwritten. Continue? (y/n)\n"
	read YN
	printf "\n"
fi

if [ "$YN" == "N" ] || [ "$YN" == "n" ]
then
	exit 1
fi

source activate dgl-env

printf "##########BEGINNING PROCESSING##########\n\n"
	
for NTUPLE in $NTUPLES
do
	printf "Running make_cuts.py to create ${DATANAME}_${NTUPLE}.hdf5 with $ENTRIES jets\n"
	python scripts/make_cuts.py -n ${DATADIR}${NTUPLE}.root -e $ENTRIES -d ${DATADIR}${DATANAME}/ -s ${DATANAME}_${NTUPLE}
	printf "\n"

	printf "Running create_graphs.py to transform ${DATANAME}_${NTUPLE}.hdf5 into GNN compatible data\n"
	python scripts/create_graphs.py -d ${DATADIR}${DATANAME}/ -s ${DATANAME}_${NTUPLE}
	printf "\n"
done

printf "Running combine_graphs.py to combine individual files\n"
python scripts/combine_graphs.py -d ${DATADIR}${DATANAME}/ -s $DATANAME -n "$NTUPLES"
	
if [[ $NORMED != 0 ]]
then
	printf "Running norm_graphs.py to normalize graphs based on training dataset\n"
	python scripts/norm_graphs.py -d ${DATADIR}${DATANAME}/ -s $DATANAME
	printf "\n"
fi