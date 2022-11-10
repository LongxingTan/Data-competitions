FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

ADD code/ /code/
COPY ./requirements.txt /requirements.txt
WORKDIR /code

RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --upgrade pip
RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r ../requirements.txt
CMD ["sh", "run.sh"]
