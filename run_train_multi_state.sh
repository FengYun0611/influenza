#!/bin/zsh
# Option
# 1 : gpu num
# 2 : predict step
# 2 : max epoch
# 3 : lr
# 4 : loss_fn
# 5 : scheduling method
# bash recipes/run_train_multi_state.sh 0 1 200 0.0001 mse
#!/bin/bash
python3 train_multi_state.py --gpu ${1} \
    --input_step 10 \
    --predict_step ${2} \
    --use_clm 0 \
    --batch_size 64 \
    --max_epoch ${3} \
    --lr ${4} \
    --outpath ./checkout/multi_state \
    --dropout_pre 0.0 \
    --dropout_post 0.0 \
    --d_model 512 \
    --ff_hidnum 1024 \
    --hid_pre 1024 \
    --hid_post 1024 \
    --loss_fn ${5}