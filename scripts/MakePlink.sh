#VcfFil=/home/local/users/jw/resources/AncestryPCA/resources/1KG.SNP.Common.Coding.vcf.gz
VcfFil=$1
OutNam=1KG.Exome.SNP.Common
BbfNam=$OutNam
SamLst=$OutNam.samples.list
touch ExcludeSNP.list 
less $VcfFil | grep -m 1 "^#CHROM" | cut -f 10- | tr '\t' '\n' > $SamLst
SamLen=`cat $SamLst | wc -l`
if [[ $SamLen -lt 500 ]]; then
    vcftools --gzvcf $VcfFil --plink --out $BbfNam.tmp
    plink --file $BbfNam.tmp --make-bed --out $BbfNam
    rm -f $BbfNam.tmp*
else
    mkdir -p TempSplit
    split --additional-suffix=.list -l 500 $SamLst $OutNam.samples.split.
    for i in $OutNam.samples.split.*.list; do
		vcftools --gzvcf $VcfFil --keep $i --exclude ExcludeSNP.list --plink --out TempSplit/${i/.list/} &
        #vcftools --gzvcf $VcfFil --keep $i --plink --out TempSplit/${i/.list/}
        plink --file TempSplit/${i/.list/} --make-bed --out TempSplit/${i/.list/}
    done
    wait
    ls TempSplit/*ped | awk -v OFS='\t' '{ $2=$1 ; gsub ( /ped$/, "map", $2) ; print }' > $BbfNam.tmp.splitlist
    plink --merge-list $BbfNam.tmp.splitlist --make-bed --out $BbfNam
    rm -rf TempSplit
fi

