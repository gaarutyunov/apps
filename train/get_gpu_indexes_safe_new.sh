#!/bin/bash

if [ -z "$SLURM_JOB_ID" ];
then
	echo "No jobs running. Exiting..."
	exit
fi
lines=$(scontrol show job $SLURM_JOB_ID -dd | grep IDX)
IFS='
'

scontrol show job $SLURM_JOB_ID -dd > scontrol_${SLURM_JOB_ID}

hostfile="hostfile_${SLURM_JOB_ID}"
rm "$hostfile"

for i in $lines
do
	for node in $(echo $i | python sanitize_node_info.py)
	do
		nodename=$(echo $node | gawk 'match($0, /Nodes=(cn-...)/,a) {print a[1]}')
		gpu_node=$(echo $node | gawk 'match($0, /gpu:(.*)\(/, a) {print a[1]}')
		echo "${nodename} slots=${gpu_node}" >> "$hostfile"
	done
done

