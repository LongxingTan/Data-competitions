import time, os
from typing import List

import ai_flow as af
from ai_flow_plugins.job_plugins import python, flink
from pyflink.table import Table
from tf_main import train
from notification_service.client import NotificationClient
from notification_service.base_notification import BaseEvent
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage


def get_dependencies_path():
    return "/opt"


class TrainModel(python.PythonProcessor):
    def __init__(self) -> None:
        super().__init__()

    def first_time(self):
        return not os.path.exists('/host/model/base_model/frozen_inference_graph.pb')

    def process(self, execution_context: python.python_processor.ExecutionContext, input_list: List) -> List:
        print('train_job triggered ')
        
        model_path = '/host/model'
        save_name = 'saved_model'
        
        if self.first_time():
            train_path = '/tcdata/train0.csv'
        else:
            train_path = '/tcdata/train1.csv'
        train(train_path, model_path, save_name)
        
        model_meta = execution_context.config['model_info']
        af.register_model_version(model=model_meta, model_path=model_path)
        model_version_meta = af.get_latest_generated_model_version(model_meta.name)
        deployed_model_version = af.get_deployed_model_version(model_name=model_meta.name)
        if deployed_model_version is not None:
            af.update_model_version(model_name=model_meta.name,
                                            model_version=deployed_model_version.version,
                                            current_stage=ModelVersionStage.DEPRECATED)
        af.update_model_version(model_name=model_meta.name,
                                        model_version=model_version_meta.version,
                                        current_stage=ModelVersionStage.VALIDATED)
        af.update_model_version(model_name=model_meta.name,
                                        model_version=model_version_meta.version,
                                        current_stage=ModelVersionStage.DEPLOYED)

        return []


class Source(flink.flink_processor.FlinkSqlProcessor):
    def open(self, execution_context: flink.ExecutionContext):
        t_env = execution_context.table_env
        t_env.get_config().set_python_executable('/opt/python-occlum/bin/python3.7')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        t_env.get_config().get_configuration().set_string("classloader.resolve-order", "parent-first")
        t_env.get_config().get_configuration().set_integer("python.fn-execution.bundle.size", 1)

    def sql_statements(self, execution_context: flink.ExecutionContext) -> List[str]:
        data_meta = execution_context.config['dataset']

        sql_statements = '''
            CREATE TABLE input_table (
                uuid STRING,
                visit_time STRING,
                user_id STRING,
                item_id STRING,
                features STRING
            ) WITH (
                'connector' = 'kafka',
                'topic' = '{}',
                'properties.bootstrap.servers' = '{}',
                'properties.group.id' = 'testGroup',
                'format' = '{}'
            )
        '''.format(data_meta.name, data_meta.uri, data_meta.data_format)
        return [sql_statements]


class Transformer(flink.flink_processor.FlinkSqlProcessor):
    def open(self, execution_context: flink.ExecutionContext):
        t_env = execution_context.table_env
        model_name = execution_context.config['model_info'].name
        model_version_meta = af.get_deployed_model_version(model_name)
        model_path = model_version_meta.model_path

        t_env.get_config().get_configuration().set_string('pipeline.global-job-parameters',
                                                          '"modelPath:""{}"""'
                                                          .format(os.path.join(model_path, 'frozen_model')))
        t_env.get_config().get_configuration().set_string("pipeline.classpaths",
                                                          "file://{}/flink-sql-connector-kafka_2.11-1.11.2.jar"
                                                          .format(get_dependencies_path()))

    def udf_list(self, execution_context: flink.ExecutionContext) -> List:
        udf_func = flink.flink_processor.UDFWrapper("cluster_serving",
                              "com.intel.analytics.zoo.serving.operator.ClusterServingFunction")
        return [udf_func]

    def sql_statements(self, execution_context: flink.ExecutionContext) -> List[str]:
        process_stmt = 'CREATE VIEW processed_table AS SELECT uuid, cluster_serving(uuid, features) AS data FROM input_table'
        return [process_stmt]


class Sink(flink.flink_processor.FlinkSqlProcessor):
    def sql_statements(self, execution_context: flink.ExecutionContext) -> List[str]:
        data_meta = execution_context.config['dataset']
        create_stmt = '''
            CREATE TABLE write_table (
                uuid STRING,
                data STRING
            ) WITH (
                'connector.type' = 'kafka',
                'connector.version' = 'universal',
                'connector.topic' = '{}',
                'connector.properties.zookeeper.connect' = '127.0.0.1:2181',
                'connector.properties.bootstrap.servers' = '{}',
                'connector.properties.group.id' = 'testGroup',
                'connector.properties.batch.size' = '1',
                'connector.properties.linger.ms' = '1',
                'format.type' = '{}'
            )
        '''.format(data_meta.name, data_meta.uri, data_meta.data_format)

        sink_stmt = 'INSERT INTO write_table SELECT * FROM processed_table'

        notification_client = NotificationClient('127.0.0.1:50051', default_namespace="default")
        notification_client.send_event(BaseEvent(key='KafkaWatcher', value='model_registered'))

        return [create_stmt, sink_stmt]
