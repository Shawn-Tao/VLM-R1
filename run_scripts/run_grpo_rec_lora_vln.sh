PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# on remote
# data_paths="/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcoco_train.jsonl:/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcocop_train.jsonl:/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcocog_train.jsonl" 
# image_folders="/training/shz/dataset/coco:/training/shz/dataset/coco:/training/shz/dataset/coco"

# data_paths="/LLM-VLM/VLM-R1/annotation/rec_jsons_processed/refcoco_train.jsonl:/LLM-VLM/VLM-R1/annotation/rec_jsons_processed/refcocop_train.jsonl:/LLM-VLM/VLM-R1/annotation/rec_jsons_processed/refcocog_train.jsonl" 
# image_folders="/LLM-VLM/VLM-R1/coco:/LLM-VLM/VLM-R1/coco:/LLM-VLM/VLM-R1/coco"

data_paths="/LLM-VLM/datasets/rl_vln/r2r_rl_train_424_240.jsonl"
image_folders="/LLM-VLM/datasets/rl_vln/r2r_train_424_240"

# model_path="/training/models/Qwen2.5-VL-3B-Instruct"
# model_path="/LLM-VLM/VLM-R1/models/Qwen2.5-VL-3B-Instruct"
model_path="/LLM-VLM/train_saves/Qwen2.5-VL-7b-424-240-9-usample-r2r/merged-950"

is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-vln-lora" # TODO: change this to your own experiment name
# TASK_TYPE="rec"
TASK_TYPE="vln"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
CUDA_VISIBLE_DEVICES=1,2

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/grpo_jsonl_vln.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 2 \
    --reward_funcs accuracy format \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 32 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --beta 0.04 \
    --bf16 false \
    --max_completion_length 128 \
    --max_length 512
    # --image_max_pixels: 101800 \
    # --video_max_pixels: 384
    # --beta 0.04
    # --use_cache False


echo "Training completed for ${EXP_NAME}"
