#!/bin/bash
# how to use it 01为版本号: sh ./docker_build.sh 01

# docker login --username=longxingtan registry.cn-shenzhen.aliyuncs.com
docker build -f Dockerfile -t registry.cn-shenzhen.aliyuncs.com/yuetan/competition:$1 .
#docker run -v ~/baseline/competition/earth/data:/tcdata --gpus all registry.cn-shenzhen.aliyuncs.com/yuetan/competition:$1
docker push registry.cn-shenzhen.aliyuncs.com/yuetan/competition:$1

echo 'Image build finished'
