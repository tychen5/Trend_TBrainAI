#! /bin/bash

root_dir=.
data_dir=${root_dir}/data
src_dir=${root_dir}/src
log_dir=${root_dir}/log

# check exception

if [ $1 == "check" ]; then
# argv[1] = root_dir
    python3 ${src_dir}/checkException.py -d $root_dir
fi
