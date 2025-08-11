Code for paper "Bridging the Capability Gap: Joint Alignment Tuning for Harmonizing LLM-based Multi-Agent Systems"




### Sample K candidate subgoal sequences from the planning agent

```bash
python Planning_Agent_Alignment/vllm_sample.py \
    --model_name ${plan_model} \
    --ds_plan ${train_data_root}/${task_name}_plan.jsonl \
    --world_size 1 \
    --dest_dir ${csv_root} \
    --output_name ${task_name}_multi_${epoch_iter}.csv 

python Planning_Agent_Alignment/subgoal_merge.py \
    --dest_dir ${csv_root} \
    --num_samples 15 \
    --ds_source ${train_data_root}/${task_name}_plan.jsonl \
    --subgoal_file ${task_name}_multi_${epoch_iter}.csv \
    --output_name ${task_name}_subgoal_${epoch_iter}.jsonl 
```


### Compute PPL-based rewards

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
    Planning_Agent_Alignment/ppl_infer_with_trainer.py \
    --model_name_or_path ${ground_model} \
    --input_file ${train_data_root}/${task_name}_ground.jsonl \
    --per_device_eval_batch_size 32 \
    --num_samples 15 \
    --subgoal_file ${csv_root}/${task_name}_subgoal_${epoch_iter}.jsonl \
    --output ${csv_root}/${task_name}_score_${epoch_iter}.csv 
```


### Training based DPO loss

```bash
python Planning_Agent_Alignment/build_compare_dpo_data.py \
    --input_score ${csv_root}/${task_name}_score_${epoch_iter}.csv \
    --input_subgoals ${csv_root}/${task_name}_multi_${epoch_iter}.csv \
    --subgoals_file ${csv_root}/${task_name}_subgoal_${epoch_iter}.jsonl \
    --ds_ground ${train_data_root}/${task_name}_ground.jsonl \
    --output ${csv_root}/dpo/dpo_${epoch_iter}.jsonl \


accelerate launch --config_file ./Planning_Agent_Alignment/acc.yaml --main_process_port 2950 Planning_Agent_Alignment/train_dpo.py \
    --model_name_or_path ${plan_model} \
    --train_data ${csv_root}/dpo/dpo_${epoch_iter}.jsonl \
    --gradient_accumulation_steps 12 \
    --gradient_checkpointing \
    --learning_rate 4e-7 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --output_dir ${model_root}/${task_name}/dpo_${epoch_iter} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 512 \
    --max_prompt_length 1024
```



### Generate the action sequences for each subgoal sequence

```bash
python Grounding_Agent_Improving/build_train_ground_data.py \
  --model_name ${ground_model} \
  --input_file ${train_data_root}/${task_name}_ground.jsonl \
  --world_size 1 \
  --subgoals_file ${task_name}_subgoal_${epoch_iter}.jsonl  \
  --score_csv ${csv_root}/${task_name}_score_${epoch_iter}.csv \
  --output ${csv_root}/sft/train_ground_${epoch_iter}.jsonl 
```

### Correct the action sequences 

```bash
python Grounding_Agent_Improving/data_process.py \
    --input_file ${csv_root}/sft/train_ground_${epoch_iter}.jsonl  \
    --output_file ${csv_root}/sft/train_ground_${epoch_iter}_corrected.jsonl 
```

### Grounding_Agent_Improving
```bash
python Grounding_Agent_Improving/generator_sft_data_prepare.py \
    --input_file ${csv_root}/sft/train_ground_${epoch_iter}_corrected.jsonl  \
    --input_file2 ${train_data_root}/${task_name}_ground.jsonl \
    --output_file ${train_sft_data_root}/train_ground_${epoch_iter}_corrected \
    --model_name ${ground_model}


DS_SKIP_CUDA_CHECK=1 deepspeed  Groundin_Agent_Improving/train.py \
    --model_name_or_path ${ground_model} \
    --train_data ${train_sft_data_root}/train_ground_${epoch_iter}_corrected \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32  \
    --bf16 \
    --deepspeed_file ./ds_cfg.json \
    --output_dir ${model_root}/${task_name} \
    --save_model sft_ground_${epoch_iter}\
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --learning_rate 2e-5
```
