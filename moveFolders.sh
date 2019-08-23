foldersPath=../videos/litter-recording/*/0_0-960_540/ #9r16c
for d in $( ls -d $foldersPath/9r*); do
#   mkdir -p ${d/../~/Desktop}
#    mv $d ${d/../~/Desktop}
	echo $d
	rm -r $d
done
