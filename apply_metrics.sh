#cd ../darknet/videos/20190111GOPR9027half-mobilenetSSD-10000-th0p7-nms0p0-iSz216
# segments=$1
# groundtruth=yolov3-litter_10000-th0p0-nms0p0-iSz608/0_0-960_540/$segments
# yoloTiny=yolov3-tiny*/0_0-960_540/$segments
# mobilenetSSD=mobilenetSSD*/0_0-960_540/$segments
videosPath=../videos/litter-recording
cd ../Object-Detection-Metrics

# for d in $(cd $videosPath; ls *.MP4); do
#     echo ${d/.MP4/}
#     gtAll=${d/.MP4/}-$groundtruth
#     echo "groundTruth: "$gtAll
#     for seg in $(cd $videosPath/$gtAll; ls -d ./*/); do
#         echo $seg
#         echo "YOLO-Tiny"
#         for det in $(cd $videosPath; ls -d ${d/.MP4/}-$yoloTiny); do
#             #echo "gt: "$videosPath/$gtAll/$seg", det: "$videosPath/$det/$seg
#             python3.7 pascalvoc.py -gt $videosPath/$gtAll/$seg -det $videosPath/$det/$seg -t 0.0001 -gtformat xyrb -detformat xyrb -sp $videosPath/$det/$seg/analysis/ -np
#         done
#         echo "MobilenetSSD"
#         for det in $(cd $videosPath; ls -d ${d/.MP4/}-$mobilenetSSD); do
#             #echo $videosPath/$det
#             python3.7 pascalvoc.py -gt $videosPath/$gtAll/$seg -det $videosPath/$det/$seg -t 0.0001 -gtformat xyrb -detformat xyrb -sp $videosPath/$det/$seg/analysis/ -np
#         done
#         #break
#     done

    
#     #break
#     #python3.7 pascalvoc.py -gt $videosPath/$groundtruth/$d/ -det $videosPath/$detections/$d/ -t 0.0001 -gtformat xyrb -detformat xyrb -sp $videosPath/$detections/$d/analysis/ -np
# done
# #
pascalvoc(){
    gtPath=$1
    detPath=$2
    echo "gt: "$gtPath", det: "$detPath
    python3.7 pascalvoc.py -gt $gtPath -det $detPath -t 0.0001 -gtformat xyrb -detformat xyrb -sp $detPath/analysis/ -np
}

time pascalvoc $videosPath/GOPR9027-yolov3-608 $videosPath/GOPR9027-mobilenet-124

# gt=yolov3-litter_10000-th0p0-nms0p0-iSz608
# mobilenetSSD=mobilenetSSD-10000-th0p5-nms0p0-iSz*
# yoloTiny=yolov3-tiny-litter_10000-th0p0-nms0p0-iSz*
# for seg in $(cd $videosPath/$gt; ls -d */); do
#     echo $seg
#     if [ "$seg" == "1r1c/" ]; then
#         for det in $(cd $videosPath; ls -d $yoloTiny); do
#             pascalvoc $videosPath/$gt/$seg $videosPath/$det/$seg
#         done
        
#         for det in $(cd $videosPath; ls -d $mobilenetSSD); do
#             pascalvoc $videosPath/$gt/$seg $videosPath/$det/$seg
#         done    
#     else
#         # echo "h"
#         for subseg in $(cd $videosPath/$gt/$seg; ls -d *_*-*_*/); do
#             for det in $(cd $videosPath; ls -d $yoloTiny); do
#                 pascalvoc $videosPath/$gt/$seg$subseg $videosPath/$det/$seg$subseg
#             done
            
#             for det in $(cd $videosPath; ls -d $mobilenetSSD); do
#                 pascalvoc $videosPath/$gt/$seg$subseg $videosPath/$det/$seg$subseg
#             done
#         done
#     fi
# done
