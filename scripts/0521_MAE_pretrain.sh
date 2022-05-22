begin=0
end=7
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0521_MAE_test_var
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

blr=1.5e-4
batch_size=512
warmup_epochs=5
# norm_pix_loss
for times in {0,1,2}; do
    export exp_args="--phase $phase"
    export run_name="warmup_epochs=$warmup_epochs,norm_pix_loss=0,phase=pretrain,blr=$blr,bs=$batch_size"
    bash base.sh --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs  
done
wait

bash 0521_MAE_linprobe_debug.sh