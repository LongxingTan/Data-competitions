FROM registry.cn-shanghai.aliyuncs.com/tcc_public/tianchi_antispam:v2

COPY /config2/flink-conf.yaml /opt/flink-1.11.2/conf/flink-conf.yaml
COPY /root/tianchi_aiflow/workflows/tianchi_main/tf_main.py /root/tianchi_aiflow/workflows/tianchi_main/tf_main.py
COPY /root/tianchi_entry/run_bash.sh /root/tianchi_entry/run_bash.sh

COPY /config2/occlum_flink_job_plugin.py /opt/python-occlum/lib/python3.7/site-packages/ai_flow_plugins/job_plugins/flink/occlum_flink_job_plugin.py
COPY /config2/occlum_flink_run_main.py /opt/python-occlum/lib/python3.7/site-packages/ai_flow_plugins/job_plugins/flink/occlum_flink_run_main.py
# COPY /config2/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.10.0-RC1-serving.jar /opt/flink-1.11.2/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.10.0-serving.jar 
