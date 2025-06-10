#!/bin/bash
# 创建虚拟环境
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# git remote set-url origin https://<your_token>@github.com/ZereChen/text_sentiment_classification.git

# 进入 src 目录
root=$(pwd)
export PYTHONPATH=$root:$PYTHONPATH
echo "当前目录是: $root"

cd src || { echo "无法进入 src 目录"; exit 1; }
$root/.venv/bin/python $root/src/train_advanced.py
