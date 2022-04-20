#!/bin/bash

# currentShellPID=$(echo $$)
# echo $currentShellPID && taskset -cp 63-75 $currentShellPID && echo $currentShellPID


declare -a basicsetupnames=(
    "N100_05X100"
)

for i in "${basicsetupnames[@]}"
do
    python3 main.py --setup=setups/${i}.json --policy=Sampled_R_UCB --L=5 > log/${i}_Sampled_R_UCB_L5.log 2>&1
done
