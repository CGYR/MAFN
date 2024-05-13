#!/bin/bash

n_gpu=1
n_cpu=6
task="gen abcn gy embs `date "+%Y-%m-%d %X"`"
while getopts "g:c:n:" arg #选项后面的冒号表示该选项需要参数
do
        case $arg in
             c)  
                n_cpu="$OPTARG"  #参数存在$OPTARG中
		        ;;
             g) 
                n_gpu="$OPTARG"
		        ;;
             n)
                task="$OPTARG"
                ;;
             ?)  #当有不认识的选项的时候arg为?
                 echo "unkonw argument"
                 exit 1
                 ;;
        esac
done

echo "gpu=${n_gpu},cpu=${n_cpu},task=${task}"


# falcon submit -T Pytorch -r /home/notebook/code/personal/appseq/src/run_aetn.py -g "${n_gpu}" -c "${n_cpu}" -n "${task}"
# falcon submit -T Pytorch -r /home/notebook/code/personal/appseq/src/childgame.py -g "${n_gpu}" -c "${n_cpu}" -n "${task}"
# -m 10240
falcon submit   -g "${n_gpu}" -c "${n_cpu}" -n "${task}" -m 10240 -r /home/notebook/code/personal/appseq/src/gen_abcn_gy_embs.py 








