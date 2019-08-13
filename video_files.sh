videosPath=../videos/litter-recording/

cd $videosPath
for d in $(ls *.MP4); do
    echo $d to ${d/.MP4/-hflip.MP4}
    ffmpeg -i $d -vf hflip -c:a copy ${d/.MP4/-hflip.MP4}
    # break
done