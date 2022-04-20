#!/bin/bash

# currentShellPID=$(echo $$)
# echo $currentShellPID && taskset -cp 0-24 $currentShellPID && echo $currentShellPID


declare -a setupnames=(
    "N100_05X100"
)

for i in "${setupnames[@]}"
do
    python3 main.py --setup=setups/${i}.json --policy=UCB > log/${i}_UCB.log 2>&1
done

