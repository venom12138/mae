begin=0
end=3
para=1
parallel_per_gpu=1
. utils.sh

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES
export MASTER_PORT=25789
export nproc_per_node=$((end - begin + 1))
export exp_name=0526_MAE_bsp_2_changelayer
export branch=master

export phase=finetune

blr=5e-4
batch_size=256 # 256*4=1024
accum_iter=1

prefix="/home/jwyu/.exp/0520_MAE_yjw/$exp_name"
export exp_args="--phase $phase --blr $blr --batch_size $batch_size --epochs 100 --dist_eval "
for bsplayer in {9,}; do
    export MASTER_PORT=25255
    export run_name="bsplayer=$bsplayer,phase=$phase,blr=$blr,bs=$batch_size"
    bash base.sh --finetune $prefix/*times=0,bsp=1,bsplayer=$bsplayer,warmup_epochs=5,phase=pretrain,*/checkpoint.pth &

    export MASTER_PORT=25795
    export run_name="bsplayer=$bsplayer,phase=$phase,blr=$blr,bs=$batch_size"
    bash base.sh --finetune $prefix/*times=1,bsp=1,bsplayer=$bsplayer,warmup_epochs=5,phase=pretrain,*/checkpoint.pth &

    # export MASTER_PORT=25872
    # export run_name="bsplayer=$bsplayer,phase=$phase,blr=$blr,bs=$batch_size"
    # bash base.sh --finetune $prefix/*times=2,bsp=1,bsplayer=$bsplayer,warmup_epochs=5,phase=pretrain,*/checkpoint.pth 
done
wait
bash 0526_linprobe_bsp_2_changelayer.sh
