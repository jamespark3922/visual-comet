#!/usr/bin/env bash

 outputDir='video_files'
 parallelDownloads=$1

 mkdir -p $outputDir

 if [ $# -lt 1 ]
 then
   parallelDownloads=8
 fi

 ########## download videos #
 wget -c https://storage.googleapis.com/ai2-mosaic/public/visualcomet/video_list.txt
 filesToDownloadMD="video_list.txt"

 echo "Downloading 105805 videos to '$outputDir'..."
 cat $filesToDownloadMD | xargs -n 1 -P $parallelDownloads wget -crnH --cut-dirs=3 -q -P $outputDir