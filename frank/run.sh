#! /bin/bash

root_dir=.
data_dir=${root_dir}/data
src_dir=${root_dir}/src
log_dir=${root_dir}/log
info_dir=${root_dir}/info

csv_dir=${info_dir}/csv
pkl_dir=${info_dir}/pkl

# check exception

if [ $1 == "checkExc" ]; then
# argv[1] = root_dir
    python3 ${src_dir}/checkException.py
elif [ $1 == "collect" ]; then
    python3 ${src_dir}/collectStatistic.py
elif [ $1 == "0" ]; then
    python3 ${src_dir}/collectStatistic2.py 0
elif [ $1 == "1" ]; then
    python3 ${src_dir}/collectStatistic2.py 1
elif [ $1 == "time" ]; then
    python3 ${src_dir}/collectTimeFeature.py
elif [ $1 == "clean" ]; then
    rm -rf ${pkl_dir}/*
    rm -rf ${csv_dir}/time_feature.csv
fi
