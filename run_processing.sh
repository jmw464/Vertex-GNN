#!/bin/bash

NTUPLES="user.jmwagner.26222492.Akt4EMPf_BTagging201903._000007 user.jmwagner.25874500.Akt4EMPf_BTagging201903._000005"
DATADIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/
DATANAME=btag_zh07_tt05_cut_v5_nopu

ENVNAME=dgl-env #name of conda environment that contains packages

ENTRIES=( 150000 150000 ) #jets used per file (after cuts)

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

source activate $ENVNAME

printf "##########BEGINNING PROCESSING##########\n\n"

file_counter=0
for NTUPLE in $NTUPLES
do
	printf "Running process_ntuple.py to create ${DATANAME}_${NTUPLE}.hdf5 with ${ENTRIES[$file_counter]} jets\n"
	python scripts/process_ntuple.py -n ${NTUPLE} -e ${ENTRIES[$file_counter]} -i ${DATADIR} -o ${DATADIR}${DATANAME}/
	printf "\n"
	file_counter=$(expr $file_counter + 1)

	printf "Running create_graphs.py to transform ${DATANAME}_${NTUPLE}.hdf5 into GNN compatible data\n"
	python scripts/create_graphs.py -d ${DATADIR}${DATANAME}/ -s ${NTUPLE}
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

python scripts/plot_data.py -d ${DATADIR}${DATANAME}/ -s $DATANAME
