RELEASE ?= 12141

.PHONY: submit
submit:
	docker build -f docker/Dockerfile -t registry.cn-shanghai.aliyuncs.com/yuetan/chain:${RELEASE} .
	docker push registry.cn-shanghai.aliyuncs.com/yuetan/chain:${RELEASE}
	echo "Image build finished"

.PHONY: localtest
localtest:
	docker kill $$(sudo docker ps -q) || echo 'Start'
	docker build -f docker/Dockerfile -t registry.cn-shanghai.aliyuncs.com/yuetan/chain:${RELEASE} .
	docker run -idt -v /home/teletraan/baseline/competition/chain/tcdata:/tcdata -p 8080:8080 registry.cn-shanghai.aliyuncs.com/yuetan/chain:${RELEASE} bash && docker exec -it $$(sudo docker ps --format '{{.ID}}') bash
	echo 'Image build finished'

.PHONY: togit
togit:
	git add .
	git commit -m ${RELEASE}
	git push

.PHONY: tolocal
tolocal:
	scp longxing@192.168.1.115:baseline/competition/chain/user_data/*.csv .
