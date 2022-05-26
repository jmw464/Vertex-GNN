#!/bin/bash
#SBATCH -n 12
#SBATCH --qos=regular
#SBATCH -C knl
#SBATCH --time=19:59:00
#SBATCH --error=proc_%j.err
#SBATCH --output=proc_%j.out

NTUPLES="user.jmwagner.27266826.Akt4EMPf_BTagging201903._000006 user.jmwagner.27266824.Akt4EMPf_BTagging201903._000006"
DATADIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/
DATANAME=btag_zh06_tt06_cut_v7_1
OPTIONFILE=options

ENVNAME=dgl-env #name of conda environment that contains packages

ENTRIES=( 300000 300000 ) #jets used per file (after cuts)

NORMED=0

#create output directory if not already there
if [ ! -d ${DATADIR}${DATANAME}/ ]
then
	mkdir ${DATADIR}${DATANAME}/
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
	printf "Running create_graphs.py to transform ${NTUPLE}.hdf5 into GNN compatible data\n\n"
	#python scripts/create_graphs.py -n ${NTUPLE} -i ${DATADIR} -o ${DATADIR}${DATANAME}/ -e ${ENTRIES[$file_counter]} -d ${DATANAME} -f options/${OPTIONFILE}
	printf "\n"
	file_counter=$(expr $file_counter + 1)
done

printf "Running combine_graphs.py to combine individual files\n\n"
#python scripts/combine_graphs.py -d ${DATADIR}${DATANAME}/ -s $DATANAME -n "$NTUPLES" -f options/${OPTIONFILE}
	
if [[ $NORMED != 0 ]]
then
	printf "Running norm_graphs.py to normalize graphs based on training dataset\n\n"
	python scripts/norm_graphs.py -d ${DATADIR}${DATANAME}/ -s $DATANAME
	printf "\n"
fi

printf "Running plot_data.py to generate plots\n\n"
#python scripts/plot_data.py -d ${DATADIR}${DATANAME}/ -s $DATANAME -f options/${OPTIONFILE}

printf "Running prune_graphs.py to remove cut tracks\n\n"
python scripts/prune_graphs.py -d ${DATADIR}${DATANAME}/ -s $DATANAME -n $NORMED
