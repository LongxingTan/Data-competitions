#!/bin/bash

${FLINK_HOME}/bin/stop-cluster.sh
stop-all-aiflow-services.sh
${KAFKA_HOME}/bin/kafka-server-stop.sh
${KAFKA_HOME}/bin/zookeeper-server-stop.sh
service mysql stop
ps -aux | grep occlum | awk '{print $2}' | xargs kill -9
ps -aux | grep aesm | awk '{print $2}' | xargs kill -9

shopt -s globstar
rm -rf /host/model
rm -rf /root/airflow
rm -rf /root/aiflow
rm -rf /root/tianchi_occlum/*.log
rm -rf /root/tianchi_occlum/model
rm -rf /root/tianchi_occlum/image/opt/*.*project
rm -rf /root/tianchi_occlum/image/opt/*.*project.zip
rm -rf /root/tianchi_aiflow/tianchi_main
rm -rf /root/tianchi_aiflow/generated
rm -f ${FLINK_HOME}/log/*
rm -f ${log_line_num} /root/tianchi_entry/*.log
rm -f ${log_line_num} /root/tianchi_entry/*.csv

# cd ~/occlum_switcher/non-occlum/
# ./switch.sh 

LD_LIBRARY_PATH=/opt/intel/sgx-aesm-service/aesm nohup /opt/intel/sgx-aesm-service/aesm/aesm_service --no-daemon >/dev/null 2>&1 &
AESM_PID=$!
cd ~/occlum_builder/
./recreate_instance.sh 
cd ~/occlum_switcher/occlum/
./switch.sh 
export START_EXTERNAL_FLINK=FALSE

# export SGX_MODE=SIM

${KAFKA_HOME}/bin/zookeeper-server-start.sh -daemon ${KAFKA_HOME}/config/zookeeper.properties
if [ ${START_EXTERNAL_FLINK}x == 'TRUE'x ]; then
    ${FLINK_HOME}/bin/start-cluster.sh
fi 
service mysql restart
mysql -e "set global explicit_defaults_for_timestamp =1;" --user=root --password=root 
mysql -e "DROP DATABASE airflow;" --user=root --password=root 
mysql -e "CREATE DATABASE airflow CHARACTER SET UTF8mb3 COLLATE utf8_general_ci;" --user=root --password=root 

sleep 20

${KAFKA_HOME}/bin/kafka-server-start.sh  -daemon ${KAFKA_HOME}/config/server.properties
start-all-aiflow-services.sh mysql://root:root@127.0.0.1/airflow

cd ~

sleep 20

time=$(date "+%Y%m%d%H%M%s")

cd ~/tianchi_entry

echo "starting workflow"

/opt/python-occlum/bin/python3.7 /root/tianchi_aiflow/workflows/tianchi_main/tianchi_main.py --server localhost:9092 --input_topic tianchi_input_${time} --output_topic tianchi_output_${time}

echo "starting workflow_util.py"

/opt/python-occlum/bin/python3.7 workflow_util.py --server localhost:9092 --input_topic tianchi_input_${time} --output_topic tianchi_output_${time} 

kill $AESM_PID

log_line_num=50
tail -n ${log_line_num} /root/aiflow/logs/*.log
tail -n ${log_line_num} /root/airflow/logs/**/*.log
tail -n ${log_line_num} /root/tianchi_occlum/*.log
tail -n 150 /root/tianchi_occlum/image/opt/*.*project/**/*.log
tail -n ${log_line_num} /root/tianchi_aiflow/workflows/tianchi_main/**/*.log
tail -n ${log_line_num} ${FLINK_HOME}/log/*
tail -n ${log_line_num} /root/tianchi_entry/*.log
head -n ${log_line_num} /root/tianchi_entry/result.csv
tail -n ${log_line_num} /root/tianchi_entry/result.csv
