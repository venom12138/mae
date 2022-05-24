begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0524_MAE_debug
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

blr=1.5e-4
batch_size=1024 # 1024*4=4096 
warmup_epochs=5

for times in {0,}; do
    export exp_args="--phase $phase "
    export run_name="times=$times,warmup_epochs=$warmup_epochs,norm_pix_loss=1,phase=pretrain,blr=$blr,bs=$batch_size"
    bash base.sh --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --norm_pix_loss
done
wait
bash 0523_MAE_linprobe_debug.sh
bash 0523_MAE_finetune.sh