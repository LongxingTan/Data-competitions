#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

instance_path=/root/tianchi_occlum

rm -rf ${instance_path}/image/opt/python-occlum
cp -r /opt/python-occlum ${instance_path}/image/opt/

./generate_hosts.sh > ${instance_path}/image/etc/hosts

mkdir -p ${instance_path}/image/tcdata/
cp /tcdata/* ${instance_path}/image/tcdata/

cp /opt/*.jar ${instance_path}/image/opt/

cp -r /opt/flink-1.11.2 ${instance_path}/image/opt/
cp -r /usr/lib/jvm/java-11-openjdk-amd64 ${instance_path}/image/usr/lib/jvm/

cp -rf /etc/ssl ${instance_path}/image/etc/
cp -rf /etc/passwd ${instance_path}/image/etc/
cp -rf /etc/group ${instance_path}/image/etc/
cp -rf /etc/java-11-openjdk ${instance_path}/image/etc/

rm -f ${instance_path}/Occlum.json
cp Occlum.json ${instance_path}/