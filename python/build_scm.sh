#!/bin/bash
set -e

ROOT_DIR=$(pwd)
PYTHON=python3
PIP="$PYTHON -m pip"
BUILD_TIME=$(date +%Y%m%d%H%M)
# 获取当前分支名，并将特殊字符转换为下划线
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
echo "BRANCH_NAME: $BRANCH_NAME"

# 如果分支是以 release_ 或 release/ 开头，则将 release_ 或 release/ 替换为空
if [[ $BRANCH_NAME =~ ^release[\/_] ]]; then
    BRANCH_NAME=${BRANCH_NAME#release}
    BRANCH_NAME=${BRANCH_NAME#/}
    BRANCH_NAME=${BRANCH_NAME#_}
    # 如果分支里还有 / ，则将 / 替换为 .
    BRANCH_NAME=${BRANCH_NAME//\//.}
    if [[ ! -z $BRANCH_NAME ]]; then
        BRANCH_NAME=.${BRANCH_NAME}
    fi
    VERSION_SUFFIX=+byted${BRANCH_NAME}.${BUILD_TIME}
elif [[ $BRANCH_NAME == ep_main ]]; then
    VERSION_SUFFIX=+iaas.dev.${BUILD_TIME}
else
    VERSION_SUFFIX=+byted.${BUILD_TIME}
fi

echo "VERSION_SUFFIX: $VERSION_SUFFIX"

TOS_UTIL_URL=https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
if [ ! -z "$CUSTOM_TOS_UTIL_URL" ]; then
    TOS_UTIL_URL=$CUSTOM_TOS_UTIL_URL
fi

cd ./python

VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)
echo "Building sglang-python version $VERSION$VERSION_SUFFIX"

pyproject_bk=pyproject.toml.bk
cp pyproject.toml $pyproject_bk

sed -i "s/^version = .*$/version = \"$VERSION$VERSION_SUFFIX\"/" pyproject.toml
$PIP install build
$PYTHON -m build

OUTPUT_PATH=$ROOT_DIR/output
mkdir -p $OUTPUT_PATH
mv dist/* $OUTPUT_PATH/
mv $pyproject_bk pyproject.toml

if [ -z "$CUSTOM_TOS_AK" ] && [ -z "$CUSTOM_TOS_SK" ]; then
    echo "CUSTOM_TOS_AK and CUSTOM_TOS_SK are not set, skip uploading to tos"
else
    # 上传制品到 tos
    wget $TOS_UTIL_URL -O tosutil && chmod +x tosutil
    for wheel_file in $(find $OUTPUT_PATH -name "*.whl"); do
        echo "uploading $wheel_file to tos..."
        ./tosutil cp $wheel_file tos://${CUSTOM_TOS_BUCKET}/packages/sglang/$(basename $wheel_file) -re cn-beijing -e tos-cn-beijing.volces.com -i $CUSTOM_TOS_AK -k $CUSTOM_TOS_SK
    done
fi
