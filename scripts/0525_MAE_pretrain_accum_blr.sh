begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0525_MAE_accumiter_blr=1e-4
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

blr=1e-4
batch_size=512 # 1024*4=4096 
warmup_epochs=5
accum_iter=2 
epochs=200

for times in {0,1,2}; do
    export exp_args="--phase $phase --accum_iter $accum_iter --norm_pix_loss --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --epochs $epochs "
    export run_name="times=$times,warmup_epochs=$warmup_epochs,norm_pix_loss=1,phase=pretrain,blr=$blr,bs=$((batch_size * accum_iter))"
    bash base.sh 
done
wait

bash 0525_MAE_linprobe_blr.sh
bash 0525_MAE_finetune_blr.sh