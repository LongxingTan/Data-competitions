FROM registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:2.4-cuda10.1-py3
# FROM registry.cn-shenzhen.aliyuncs.com/yuetan/competition:1
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3


## 把当前文件夹里的文件构建到镜像的根目录下, 注意 . 和 / 间是有个空格的
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /code

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip \
#     && \

RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --upgrade pip
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r ../requirements.txt
RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r ../requirements.txt

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]


# # Create a virtualenv for dependencies. This isolates these packages from
# # system-level packages.
# # Use -p python3 or -p python3.7 to select python version. Default is version 2.
# RUN virtualenv /env -p python3

# # Setting these environment variables are the same as running
# # source /env/bin/activate.
# ENV VIRTUAL_ENV /env
# ENV PATH /env/bin:$PATH

# RUN apt-get update && apt-get install -y python-opencv

# # Copy the application's requirements.txt and run pip to install all
# # dependencies into the virtualenv.
# ADD requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt

# # Add the application source code.
# ADD . /app

# # Run a WSGI server to serve the application. gunicorn must be declared as
# # a dependency in requirements.txt.
# CMD gunicorn -b :$PORT main:app


