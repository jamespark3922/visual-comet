#!/usr/bin/env bash

 outputDir='video_files'
 parallelDownloads=$1

 mkdir -p $outputDir

 if [ $# -lt 1 ]
 then
   parallelDownloads=8
 fi

 ########## download videos #
 wget -c https://storage.googleapis.com/jamesp/lsmdc/video_list.txt
 filesToDownloadMD="video_list.txt"

 echo "Downloading videos..."
 cat $filesToDownloadMD | xargs -n 1 -P $parallelDownloads wget -crnH --cut-dirs=3 -q -P $outputDir