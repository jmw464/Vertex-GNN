#!/bin/bash

NTUPLES="user.jmwagner.25874500.Akt4EMPf_BTagging201903._000005" #"user.jmwagner.25874500.Akt4EMPf_BTagging201903._000005"
DATADIR=/global/cfs/cdirs/atlas/jmw464/data/
DATANAME=btag_05_19_cut_v4.1c #btag_05_19_cut_v6c

ENTRIES=150000 #jets used per file (after cuts)

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
	#python scripts/make_cuts.py -n ${DATADIR}${NTUPLE}.root -e $ENTRIES -d ${DATADIR}${DATANAME}/ -s ${DATANAME}_${NTUPLE}
	printf "\n"

	printf "Running create_graphs.py to transform ${DATANAME}_${NTUPLE}.hdf5 into GNN compatible data\n"
	#python scripts/create_graphs.py -d ${DATADIR}${DATANAME}/ -s ${DATANAME}_${NTUPLE}
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
