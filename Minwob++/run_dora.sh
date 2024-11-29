export CUDA_VISIBLE_DEVICES=0
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
model_hf_dir='lmsys/vicuna-7b-v1.3'

datasets=(click-menu)

for i in ${datasets[@]}; do
    echo $i
    python DORA.py \
    --task $i \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --model_hf_dir ${model_hf_dir} \
done