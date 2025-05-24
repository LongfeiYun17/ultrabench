#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
CONFIG="configs/eval.yaml"

# TYPE
TYPE="hf"
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.95
REPETITION_PENALTY=1.0
EVAL_MODEL=gpt-4.1-mini
VERIFY_TEMPLATE=prompt/verify.md
DATA_PATH=data/fineweb_200000_test.jsonl
EVAL_NUM_PROCESSES=16

HF_MODELS=(
    meta-llama/Llama-3.1-8B-Instruct
)

API_MODELS=(
    gpt-4o
)

# GPT script
if [ $TYPE == "api" ]; then
    for model in ${API_MODELS[@]}; do
        python baselines/zero_shot/run.py \
            --type api \
            --config $CONFIG \
            --model_name $model \
            --max_new_tokens $MAX_NEW_TOKENS \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --repetition_penalty $REPETITION_PENALTY \
            --test_data $DATA_PATH \
            --eval_model $EVAL_MODEL \
            --verify_template $VERIFY_TEMPLATE \
            --num_processes $EVAL_NUM_PROCESSES
    done
fi

# HuggingFace script
if [ $TYPE == "hf" ]; then
    for model in ${HF_MODELS[@]}; do
        python baselines/zero_shot/run_sglang.py \
            --config $CONFIG \
            --model_name $model \
            --max_new_tokens $MAX_NEW_TOKENS \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --repetition_penalty $REPETITION_PENALTY \
            --test_data $DATA_PATH \
            --eval_model $EVAL_MODEL \
            --verify_template $VERIFY_TEMPLATE \
            --num_processes $EVAL_NUM_PROCESSES
    done
fi
