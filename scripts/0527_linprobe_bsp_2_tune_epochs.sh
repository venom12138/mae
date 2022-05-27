begin=0
end=1
para=1
parallel_per_gpu=1

export nproc_per_node=2 # $((end - begin + 1))
export branch=master
export phase=linprobe
export exp_name=0527_MAE_bsp_2_tune_epochs
export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh
blr=0.1
batch_size=8192
accum_iter=1
prefix="/home/jwyu/.exp/0520_MAE_yjw/$exp_name"

export exp_args="--cls_token --phase $phase --blr $blr --batch_size $batch_size --epochs 90 --dist_eval "

for epochs0 in {100,125,150,175}; do
    export MASTER_PORT=25755
    export run_name="epochs=$epochs0+$((200-epochs0)),phase=$phase,blr=$blr,bs=$batch_size"
    CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune $prefix/*times=0,epochs=$((200-epochs0)),bsp=1,warmup_epochs=5,phase=pretrain,*/checkpoint.pth &

    export MASTER_PORT=25759
    export run_name="epochs=$epochs0+$((200-epochs0)),phase=$phase,blr=$blr,bs=$batch_size"
    CUDA_VISIBLE_DEVICES=2,3 bash base.sh --finetune $prefix/*times=1,epochs=$((200-epochs0)),bsp=1,warmup_epochs=5,phase=pretrain,*/checkpoint.pth
    wait
    export MASTER_PORT=25783
    export run_name="epochs=$epochs0+$((200-epochs0)),phase=$phase,blr=$blr,bs=$batch_size"
    CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune $prefix/*times=2,epochs=$((200-epochs0)),bsp=1,warmup_epochs=5,phase=pretrain,*/checkpoint.pth &
done
wait
