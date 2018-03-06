#! /bin/bash

root_dir=.
data_dir=${root_dir}/data
src_dir=${root_dir}/src
log_dir=${root_dir}/log

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

fi
