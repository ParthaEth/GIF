#!/bin/bash

if [[ $1 -eq 0 ]] ; then
    echo 'Run id to clear must be provided. Wont clear anything. Goodbye.'
    exit 0
fi

echo "clearing dirs for run $1"

smpl_dir="./sample/$1/*.png"
echo "clearing $smpl_dir"
rm -rf $smpl_dir
echo "<========================= done ======================>"

chk_pt_dir="./checkpoint/$1/*"
echo "clearing $chk_pt_dir"
rm -rf $chk_pt_dir
echo "<========================= Done ======================>"