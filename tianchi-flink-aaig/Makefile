# use: make submit RELEASE=10111
RELEASE ?= 10111

# build container for submit
.PHONY: submit
submit:
	docker build -f config/Dockerfile -t registry.cn-shanghai.aliyuncs.com/yuetan/aaig:${RELEASE} .	
	docker push registry.cn-shanghai.aliyuncs.com/yuetan/aaig:${RELEASE}
	echo 'Image build finished'

# localtest: docker version
.PHONY: localtest
localtest:
	docker kill $$(sudo docker ps -q) || echo 'Start'
	docker build -f config/Dockerfile -t registry.cn-shanghai.aliyuncs.com/yuetan/aaig:${RELEASE} .
	docker run -idt -v /home/tlx/competition/AAIG/tcdata:/tcdata registry.cn-shanghai.aliyuncs.com/yuetan/aaig:${RELEASE} bash && docker exec -it $$(sudo docker ps --format '{{.ID}}') bash
	echo 'Image build finished'

# localtest2: local version
.PHONY: localtest2
localtest2:	
	rm -r ./weights/ || echo 'Start'
	cd root/tianchi_aiflow/workflows/tianchi_main && python tf_main.py

# recall result from container to local
.PHONY: recall
recall:
	docker cp $$(sudo docker ps --format '{{.ID}}'):/root/tianchi_entry/result.csv ./user_data/result.csv
	docker cp $$(sudo docker ps --format '{{.ID}}'):/root/tianchi_entry/nohup.out ./user_data/nohup.out

# push to github
.PHONY: togit
togit:
	git add .
	git commit -m ${RELEASE}
	git push
