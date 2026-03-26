#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

seq_len=60
label_len=24

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_path="${SCRIPT_DIR}/../dataset/"


models=(
    "DST-Mamba"
    "TFEGRU"
    "iTransformer"
    "S-Mamba"
    "PatchTST"
    "FEDformer"
    "TimesNet"
    "Dlinear"
)


datasets=(
    # "ali17.csv:cpu"
    # "ali18.csv:cpu"
    # "ali20.csv:cpu"
    # "gc11.csv:cpu"
    "gc19.csv:cpu"
)

# datasets=(
#     # "ali17.csv:mem"
#     # "ali18.csv:mem"
#     # "ali20.csv:mem"
#     # "gc11.csv:mem"
#     "gc19.csv:mem"
# )


# ===============================
# pred_len
# ===============================
# pred_lens=(24 48 72 96)
pred_lens=(48)


get_feature_dim() {
    file=$1
    col_num=$(head -n 1 $file | awk -F',' '{print NF}')
    echo $((col_num - 1))
}


for model_name in "${models[@]}"; do
for item in "${datasets[@]}"; do

    data_path="${item%%:*}"
    target="${item##*:}"
    full_path="${root_path}${data_path}"

    enc_in=$(get_feature_dim $full_path)
    dec_in=$enc_in
    c_out=1

    echo "==============================="
    echo "Model  : $model_name"
    echo "Dataset: $data_path"
    echo "Dim    : $enc_in"
    echo "==============================="

    for pred_len in "${pred_lens[@]}"; do

        if [ $pred_len -le 48 ]; then
            e_layers=2
            lr=0.005
        else
            e_layers=3
            lr=0.001
        fi

        extra_args=""

        if [ "$model_name" == "iTransformer" ]; then
            extra_args="--n_heads 8"
        fi

        if [ "$model_name" == "PatchTST" ]; then
            extra_args="--patch_len 16 --stride 8"
        fi

        if [ "$model_name" == "Crossformer" ]; then
            extra_args="--seg_len 12"
        fi

        if [ "$model_name" == "TimesNet" ]; then
            extra_args="--top_k 5"
        fi

        python -u run_longExp.py \
            --is_training 1 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id ${data_path%.*}_${model_name}_sl${seq_len}_pl${pred_len} \
            --model $model_name \
            --data custom \
            --features M \
            --freq h \
            --target $target \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --e_layers $e_layers \
            --d_layers 1 \
            --factor 3 \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --d_model 128 \
            --d_ff 128 \
            --dropout 0.1 \
            --batch_size 64 \
            --learning_rate $lr \
            --itr 1 \
            $extra_args

    done
done
done