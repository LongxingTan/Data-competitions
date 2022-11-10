# use:  make submit RELEASE=0824.1
RELEASE ?= 0.0.1

# docker login --username=longxingtan registry.cn-shenzhen.aliyuncs.com
# docker run -idt -v /Users/lx/Repository/Competition/ENSO/tcdata:/tcdata registry.cn-shanghai.aliyuncs.com/yuetan/aaig:${RELEASE} bash
submit:
	docker build -f Dockerfile -t registry.cn-shenzhen.aliyuncs.com/yuetan/competition:${RELEASE} .	
	docker push registry.cn-shenzhou.aliyuncs.com/yuetan/competition:${RELEASE}
	echo 'Image build finished'
