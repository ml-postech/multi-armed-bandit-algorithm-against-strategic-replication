#!/bin/bash

# currentShellPID=$(echo $$)
# echo $currentShellPID && taskset -cp 25-49 $currentShellPID && echo $currentShellPID


declare -a setupnames=(
    "N100_05X100"
)

for i in "${setupnames[@]}"
do
    python3 main.py --setup=setups/${i}.json --policy=H_UCB > log/${i}_H_UCB.log 2>&1
done

# python3 main.py --setup=setups/N100_05X100.json --policy=UCB