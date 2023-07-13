#!/bin/bash

chr=8
dir=/data/robin/EBB_server/OUTPUTS/Imputed_for_Matthew/chunks
nchunks=15
nstart=1
echo $dir

module load bcftools

for (( i = $nstart; i <= $nchunks; i++ )); do
    inname=$dir/chr$chr/chr$chr.chunk_0$i.trios_duos.imputed.bcf
    outname=$dir/chr$chr/chr$chr\_chunk$i\_POO\_filtered.vcf.gz
    if [ $i -lt 10 ]; then 
        inname=$dir/chr$chr/chr$chr.chunk_00$i.trios_duos.imputed.bcf
        outname=$dir/chr$chr/chr$chr\_chunk$i\_POO\_filtered.vcf.gz
    fi
    echo $inname
    echo $outname
    bcftools filter -i 'INFO > 0.9 && AF > 0.05 && FILTER="PASS"' $inname -O z -o $outname
    bcftools index $outname
    bcftools index -n $outname >> markers_chr$chr.txt 
done
