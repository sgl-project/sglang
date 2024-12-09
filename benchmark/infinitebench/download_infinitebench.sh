# reference: https://github.com/OpenBMB/InfiniteBench/blob/51d9b37b0f1790ead936df2243abbf7f0420e439/scripts/download_dataset.sh

save_dir=$1
mkdir -p ${save_dir}

for file in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
    wget -c https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/${file}.jsonl?download=true -O ./${save_dir}/${file}.jsonl
done
