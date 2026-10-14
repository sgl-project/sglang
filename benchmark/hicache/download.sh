#!/usr/bin/bash

# The usage function
usage() {
    echo "Usage: $0 {sharegpt|ultragpt|loogle|nextqa|all}"
    exit 1
}

# The download function
download() {
    case "$1" in
        sharegpt)
            echo $1
            wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
            ;;
        ultragpt)
            echo $1
            # Questions about the world
            wget https://cloud.tsinghua.edu.cn/seafhttp/files/be1d7b87-22ca-449e-a6a7-c61d1ea7e010/ultrachat_release_230407.json
            # Writing and Creation
            wget https://cloud.tsinghua.edu.cn/seafhttp/files/61742d2a-25e2-4d08-b2b9-15f47ae50ace/ultrachat_material_release_230417.json
            wget https://cloud.tsinghua.edu.cn/seafhttp/files/f71f6aa6-d346-4b16-85b7-8502efa3d608/ultrachat_material_release_230412.json
            # External materials
            wget https://cloud.tsinghua.edu.cn/seafhttp/files/42d22e28-e899-4975-a70f-5eda163e265d/ultrachat_existent_material_release_230420.json.gz
            gunzip ultrachat_existent_material_release_230420.json.gz
            ;;
        loogle)
            echo $1
            git lfs install
            git clone git@hf.co:datasets/bigainlco/LooGLE
            unzip LooGLE/data.zip
            ;;
        nextqa)
            echo $1
            git lfs install
            git clone https://huggingface.co/datasets/lmms-lab/NExTQA
            unzip NExTQA/videos.zip
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Arg check
if [ "$#" -ne 1 ]; then
    usage
fi

# Invoke

case "$1" in
    sharegpt|ultragpt|loogle|nextqa)
        download "$1"
        ;;
    all)
        download sharegpt
        download ultragpt
        download loogle
        download nextqa
        ;;
    *)
        usage
        ;;
esac
