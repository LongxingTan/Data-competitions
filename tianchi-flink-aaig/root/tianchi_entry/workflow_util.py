import csv
import os
import time, threading
from subprocess import Popen
from typing import Dict
from uuid import uuid1
import pandas as pd
import yaml
from notification_service.client import NotificationClient
from notification_service.base_notification import EventWatcher, BaseEvent
from kafka import KafkaProducer, KafkaAdminClient, KafkaConsumer
from kafka.admin import NewTopic
from typing import List
import sys, getopt
import json


def init_kafka(bootstrap_servers, input_topic, output_topic):
    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    topics = admin_client.list_topics()
    if input_topic not in topics:
        print("create input topic: "+input_topic)
        admin_client.create_topics(
            new_topics=[NewTopic(name=input_topic, num_partitions=1, replication_factor=1)])
    if output_topic not in topics:
        print("create output topic: "+output_topic)
        admin_client.create_topics(
            new_topics=[NewTopic(name=output_topic, num_partitions=1, replication_factor=1)])


def push_kafka(bootstrap_servers, input_filename, input_topic):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v.encode())
    
    f = open(input_filename)
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        producer.send(input_topic,value=line)
        time.sleep(0.004)  # 每4ms 提交一个样本
    time.sleep(10)


def listen_kafka(bootstrap_servers, output_filename, input_topic, output_topic):
    consumer = KafkaConsumer(
        input_topic, 
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        consumer_timeout_ms=1000
    )
    input_time = {}
    for message in consumer:
        input_time[int(message.value.decode().split(',')[0])] = message.timestamp
    print('received ' + str(len(input_time)) + ' messages from input topic.')
    time.sleep(10)

    consumer = KafkaConsumer(
        output_topic, 
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        consumer_timeout_ms=1000
    )
    output_time = {}
    output_label = {}
    for message in consumer:
        line = message.value.decode().strip()
        uid = int(line.split(',')[0])
        output_time[uid] = message.timestamp
        output_label[uid] = int(json.loads(','.join(line.split(',')[1:])[1:-1].replace('""','"'))['data'][0])
    print('received ' + str(len(output_time)) + ' messages from output topic.')
    
    resultf = open(output_filename, 'w+')
    for uid in input_time:
        if uid not in output_label or uid not in output_time:
            continue
        resultf.writelines(['{},{},{},{}\n'.format(uid, input_time[uid], output_time[uid], output_label[uid])])
    print('kafka messages have been written to ' + output_filename)


def send_timed_event():
    time.sleep(60 * 15)  # 首轮预测数据发送后，最迟15分钟后会发送第二批预测数据
    notification_client = NotificationClient('127.0.0.1:50051', default_namespace="default")
    notification_client.send_event(BaseEvent(key='KafkaWatcher', value='model_registered'))


class KafkaWatcher(EventWatcher):
    def __init__(self, bootstrap_servers, input_topic, output_topic):
        super().__init__()
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.count = 0
        self.t = threading.Thread(target=send_timed_event, name='send_timed_event')
        self.t.daemon = True

    def process(self, events: List[BaseEvent]):
        print("watcher event triggered " + str(self.count))
        time.sleep(20)
        if self.count == 0:
            push_kafka(self.bootstrap_servers, '/tcdata/predict0.csv', self.input_topic)
            listen_kafka(self.bootstrap_servers, './result.csv', self.input_topic, self.output_topic)
            notification_client = NotificationClient('127.0.0.1:50051', default_namespace="default")
            notification_client.send_event(BaseEvent(key='train_job', value='start'))
            self.t.start()
        else:
            push_kafka(self.bootstrap_servers, '/tcdata/predict1.csv', self.input_topic)
            listen_kafka(self.bootstrap_servers, './result.csv', self.input_topic, self.output_topic)
            sys.exit()
        print("watcher event finished " + str(self.count))
        self.count += 1
        

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],"",["input_topic=","output_topic=","server="])  
    mydict = dict(opts)
    input_topic = mydict.get('--input_topic', '')
    output_topic = mydict.get('--output_topic', '')
    bootstrap_servers = mydict.get('--server', '')
    bootstrap_servers = bootstrap_servers.split(',')

    init_kafka(bootstrap_servers, input_topic, output_topic)

    notification_client = NotificationClient('localhost:50051', default_namespace="default")
    notification_client.start_listen_event(key='KafkaWatcher', event_type='UNDEFINED', namespace="default",
        watcher=KafkaWatcher(bootstrap_servers, input_topic, output_topic),
        start_time=0)
