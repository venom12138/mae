if [ -z "$debug" ]; then
   if [ -z "$cluster" ]
   then
      bash entry.sh "${@}"
   else
      srun -J lw -N 1 -p RTX2080Ti --gres gpu:1 bash entry.sh "${@}"
   fi
else
   bash entry.sh "${@}"
fi
