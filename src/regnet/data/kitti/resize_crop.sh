#!/bin/sh

#process all depth images
echo "processing rgb images..."
for file in $(find ./data/external -type f -exec file {} \; | awk '/image_02/' | awk -F: '{if ($2 ~/image/) print $1}'); do
    out=$(echo $file | sed "s/external/processed/g" | sed "s:image_02/data:rgb:g")
    mkdir -p "$(dirname "$out")"
    echo "$out"
    convert -resize 1696x512 -interpolate bilinear -gravity center -crop 1392x512+0+0 $file $out
    done
echo "\n"

#process all depth images
echo "processing depth maps..."
for file in $(find ./data/interim -type f -exec file {} \; | awk -F: '{if ($2 ~/image/) print $1}'); do
    out=$(echo $file | sed "s/interim/processed/g" | sed "s/depth_maps/depth/g")
    mkdir -p "$(dirname "$out")"
    echo "$out"
    convert -resize 1696x512 -interpolate bilinear -gravity center -crop 1392x512+0+0 $file $out
    done
echo "done"

# mogrify -resize 1242x376! *.png
# ffmpeg -framerate 25 -i %010d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
