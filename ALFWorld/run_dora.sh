export CUDA_VISIBLE_DEVICES=0
SFT=5
INTRINSIC_DIM=10
model_dir='lmsys/vicuna-7b-v1.3'


datasets=(pick_and_place pick_cool_then_place pick_heat_then_place pick_two_obj pick_clean_then_place look_at_obj)

for i in ${datasets[@]}; do
    echo $i
    python DORA.py \
    --task $i \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
done