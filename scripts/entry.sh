#!/bin/bash
if [ -z "$debug" ]; then
  #!/bin/bash
  export WANDB_PROJECT=0520_MAE_yjw
  export WANDB_DIR="$HOME/.exp/0520_MAE_yjw/wandb/$exp_name"
  mkdir -p $WANDB_DIR

  # cd "../git_repo"

  # [ -z "$branch" ] && branch=master
  # [ -z "$commit" ] && commit=$branch
  # git checkout -f $commit
  # git reset --hard HEAD
  git add .
  git commit -m 'savecode'
  # ln -sf ../data .

  export commit=$(git rev-parse HEAD)
  echo checkout commit $commit
  sleep 5
else
  export WANDB_MODE='offline'
fi

if test "$phase" = pretrain
then
  entry_file=main_pretrain.py
elif test "$phase" = linprobe
then
  entry_file=main_linprobe.py
elif test "$phase" = finetune
then
  entry_file=main_finetune.py
fi

export MASTER_ADDR='localhost'
export MASTER_PORT=25789

[ -z "$entry_file" ] && entry_file=main_pretrain.py
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port $MASTER_PORT $entry_file --en_wandb\
 $exp_args "${@}"

sleep 5