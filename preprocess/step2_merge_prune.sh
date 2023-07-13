#!/bin/bash

chr=8
dir=/data/robin/EBB_server/OUTPUTS/Imputed_for_Matthew/chunks/chr$chr
nchunks=15
nstart=2
r2=0.9
echo $dir
outname=$dir/chr$chr\_POO\_fwor2.vcf.gz

module load bcftools

for (( i = $nstart; i <= $nchunks; i++ )); do
    inname+=$dir/chr$chr\_chunk$i\_POO\_filtered.vcf.gz
    inname+=" "
done
echo $inname
echo $outname

file=$dir/chr$chr\_POO\_filtered.vcf.gz
# merge datasets
bcftools concat $inname -Oz -o $outname
# cleanup files
rm chr$chr/chr$chr\_chunk*

# prune LD: remove all columns with R2 > $r2
bcftools +prune -m r2=$r2 -w 500kb $outname -Oz -o $file
# cleanup file
rm $outname

file1=$dir/chr$chr\_POO\_filtered_na.vcf.gz
# add missing column  F_MISSING = fraction of samples with missing genotype / per site ->  filters columns
bcftools +fill-tags $file -o $file1 -- -t F_MISSING
rm $file
#filter on missing data: remove columns with more than 5% NA
bcftools filter -e 'F_MISSING > 0.05' $file1 -o $file
# cleanup
rm $file1

# get number of markers
bcftools index $file
bcftools index -n $file >> numbers\_chr$chr.txt 
