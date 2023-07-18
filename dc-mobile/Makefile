V ?= 07201

.PHONY: pyenv
pyenv:
	pyenv activate comp


.PHONY: initgit
initgit:
	git init
	git add .
	git commit -m "initial"
	# git remote rm origin
	git remote add origin "git@github.com:yuetan1988/Mobile2022.git"
	git push -u origin master


.PHONY: togit
togit:
	# update the code and trigger the github action
	git add .
	git commit -m ${V}
	git push


.PHONY: tolocal
tolocal:
	scp -r yuetan@192.168.1.115:competition/mobile/docs/*.zip .


.PHONY: sub
sub:
	cd ./weights && zip -r ../docs/26_旋转机械.zip ./*
