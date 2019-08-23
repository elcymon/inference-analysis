network=$1 #mobilenetSSD-10000-th0p5-nms0p0-iSz220
vidFolder=../videos/litter-recording
detectionDestination=$network

for src in $(cd $vidFolder; ls -d *-$network); do
    #handle 1r1c
    echo $src": 1r1c"
    mkdir -p $vidFolder/$detectionDestination/1r1c/analysis
    mv $vidFolder/$src/0_0-960_540/*.txt $vidFolder/$detectionDestination/1r1c
    
    segFolder=3r4c
    echo $src": "$segFolder
    for seg in $(cd $vidFolder/$src/0_0-960_540/$segFolder; ls -d *_*-*_*); do
        echo "    "$seg
        #create destination folder
        mkdir -p $vidFolder/$detectionDestination/$segFolder/$seg/analysis
        
        #move contents across
        mv $vidFolder/$src/0_0-960_540/$segFolder/$seg/*.txt $vidFolder/$detectionDestination/$segFolder/$seg

    done

    segFolder=5r9c
    echo $src": "$segFolder
    for seg in $(cd $vidFolder/$src/0_0-960_540/$segFolder; ls -d *_*-*_*); do
        echo "    "$seg
        #create destination folder
        mkdir -p $vidFolder/$detectionDestination/$segFolder/$seg/analysis
        
        #move contents across
        mv $vidFolder/$src/0_0-960_540/$segFolder/$seg/*.txt $vidFolder/$detectionDestination/$segFolder/$seg

    done
    
done
# 
# mkdir $vidFolder/$detectionDestination/analysis