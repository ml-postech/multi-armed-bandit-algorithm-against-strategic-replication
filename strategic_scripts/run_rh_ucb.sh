#!/bin/bash

# currentShellPID=$(echo $$)
# echo $currentShellPID && taskset -cp 50-62 $currentShellPID && echo $currentShellPID


declare -a setupnames=(
    "N100_05X100"
)

for i in "${setupnames[@]}"
do
    python3 main.py --setup=setups/${i}.json --policy=RH_UCB > log/${i}_RH_UCB.log 2>&1
done
