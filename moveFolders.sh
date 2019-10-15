# foldersPath=../videos/litter-recording/*/0_0-960_540/ #9r16c
# for d in $( ls -d $foldersPath/9r*); do
# #   mkdir -p ${d/../~/Desktop}
# #    mv $d ${d/../~/Desktop}
# 	echo $d
# 	rm -r $d
# done

src=/media/elcymon/phdBackup/litter-detection/videos/litter-recording/$1
cd $src
dst=/home/elcymon/litter-detection/videos/litter-recording/$2
for d in $( ls -d [1-5]r[1-9]c ); do
	echo $d
	if [ "$d" == "1r1c" ]; then
		# echo "   done"
		cp $d/*GOPR9027-0*.txt $dst/$d/
	else
		for seg in $( cd $d; ls -d [0-960]* ); do
			echo "   $seg"
			mkdir -p $dst/$d/$seg && cp $d/$seg/*GOPR9027-0*.txt $dst/$d/$seg
		done
	fi
done