#cd ../darknet/videos/20190111GOPR9027half-mobilenetSSD-10000-th0p7-nms0p0-iSz216
groundtruth=20190111GOPR9027-yolov3-litter_10000-th0p1-nms0p0-iSz608-seg18r32c #20190111GOPR9027-yolov3-litter_10000-th0p1-nms0p0-iSz608-seg3r4c
detections=20190111GOPR9027half-mobilenetSSD-10000-th0p7-nms0p0-iSz216-seg18r32c #20190111GOPR9027half-yolov3-tiny-litter_10000_216-th0p1-nms0p0-iSz216-seg18r32c #20190111GOPR9027half-mobilenetSSD-10000-th0p7-nms0p0-iSz216-seg3r4c #20190111GOPR9027half-yolov3-tiny-litter_10000_216-th0p1-nms0p0-iSz216-seg3r4c #
videosPath=/home/elcymon/litter-detection/darknet/videos
cd ../Object-Detection-Metrics

for d in $(cd $videosPath/$detections; ls -d *_*-*_*); do
    echo "$d"
    python3.7 pascalvoc.py -gt $videosPath/$groundtruth/$d/ -det $videosPath/$detections/$d/ -t 0.01 -gtformat xyrb -detformat xyrb -sp $videosPath/$detections/$d/analysis/ -np
done