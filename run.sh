epoch_iter=2
task_name=llama_web_agent_15
export CUDA_VISIBLE_DEVICES=6,7


train_data_root=/home/shizhengliang/MOAT/llumos_data/train/web_agent

ground_model=/data/szl/MOAT_model/lumos_web_agent_ground_iterative
plan_model=/data/szl/MOAT_model/lumos_web_agent_plan_iterative

csv_root=/data/szl/MOAT_model/llama/llama_output/${task_name}
model_root=/data/szl/MOAT_model/llama/web_agent_model/web_agent/${task_name}
train_sft_data_root=/data/szl/MOAT_model/llama/llama_output/${task_name}

# python Planning_Agent_Alignment/vllm_sample.py \
#     --model_name ${plan_model} \
#     --ds_plan ${train_data_root}/lumos_web_agent_plan_iterative_onetime.jsonl \
#     --world_size 1 \
#     --dest_dir ${csv_root} \
#     --output_name ${task_name}_multi_${epoch_iter}.csv 

# python Planning_Agent_Alignment/subgoal_merge.py \
#     --dest_dir ${csv_root} \
#     --num_samples 15 \
#     --ds_source ${train_data_root}/lumos_web_agent_plan_iterative_onetime.jsonl \
#     --subgoal_file ${csv_root}/${task_name}_multi_${epoch_iter}.csv  \
#     --output_name ${task_name}_fab_${epoch_iter}.jsonl

# torchrun --nnodes=1 --nproc_per_node=1 --master_port=25935 \
#     Planning_Agent_Alignment/ppl_infer_with_trainer.py \
#     --model_name_or_path ${ground_model} \
#     --input_file ${train_data_root}/lumos_web_agent_ground_iterative_onetime.jsonl \
#     --per_device_eval_batch_size 32 \
#     --num_samples 15 \
#     --subgoal_file ${csv_root}/${task_name}_fab_${epoch_iter}.jsonl \
#     --output ${csv_root}/${task_name}_score_${epoch_iter}.csv 

# python Planning_Agent_Alignment/chai.py \
#     --input_multi ${csv_root}/${task_name}_multi_${epoch_suffix}.csv \
#     --ds_ground ${train_data_root}/lumos_web_agent_ground_iterative_onetime.jsonl \
#     --ds_plan ${train_data_root}/lumos_web_agent_plan_iterative_onetime.jsonl \
#     --output_multi ${csv_root}/${task_name}_multi_de_${epoch_suffix}.csv

# python attacker_dpo/build_compare_dpo_data.py \
#     --input_score ${csv_root}/${task_name}_score_${epoch_suffix}.csv \
#     --input_docs ${csv_root}/${task_name}_multi_de_${epoch_suffix}.csv \
#     --ds_ground ${train_data_root}/lumos_web_agent_ground_iterative_onetime.jsonl \
#     --output ${csv_root}/attacker_train_fab_0/dpo/dpo_0.1_${epoch_suffix}.jsonl \
#     --fab_file ${csv_root}/${task_name}_fab_${epoch_suffix}.jsonl

# python Planning_Agent_Alignment/build_compare_dpo_data.py \
#     --input_score ${csv_root}/${task_name}_score_${epoch_iter}.csv \
#     --input_subgoals ${csv_root}/${task_name}_multi_de_${epoch_iter}.csv \
#     --subgoals_file ${csv_root}/${task_name}_subgoal_${epoch_iter}.jsonl \
#     --ds_ground ${train_data_root}/lumos_web_agent_ground_iterative_onetime.jsonl  \
#     --output ${csv_root}/dpo/dpo_${epoch_iter}.jsonl \


# accelerate launch --config_file ./Planning_Agent_Alignment/acc.yaml --main_process_port 2950 Planning_Agent_Alignment/train_dpo.py \
#     --model_name_or_path ${plan_model} \
#     --train_data ${csv_root}/dpo/dpo_${epoch_iter}.jsonl \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --learning_rate 4e-7 \
#     --lr_scheduler_type cosine \
#     --num_train_epochs 1 \
#     --output_dir ${model_root}/${task_name}/dpo_${epoch_iter} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --max_length 512 \
#     --max_prompt_length 1024

# python Grounding_Agent_Improving/build_train_ground_data.py \
#   --model_name ${ground_model} \
#   --input_file ${train_data_root}/lumos_web_agent_ground_iterative_onetime.jsonl  \
#   --world_size 1 \
#   --subgoals_file ${csv_root}/${task_name}_subgoal_${epoch_iter}.jsonl  \
#   --score_csv ${csv_root}/${task_name}_score_${epoch_iter}.csv \
#   --output ${csv_root}/sft/train_ground_${epoch_iter}.jsonl 


# python Grounding_Agent_Improving/data_process.py \
#     --input_file ${csv_root}/sft/train_ground_${epoch_iter}.jsonl  \
#     --output_file ${csv_root}/sft/train_ground_${epoch_iter}_corrected.jsonl 


python reward_model/data_process_split.py \
    --input_file ${csv_root}/attacker_train_fab_0/sft/train_ground_${epoch_suffix}_llama_32b.jsonl  \
    --output_file ${csv_root}/attacker_train_fab_0/sft/train_ground_${epoch_suffix}_llama_good_32b.jsonl 