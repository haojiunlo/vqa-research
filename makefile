ECR_URL = 259628828801.dkr.ecr.ap-southeast-1.amazonaws.com
ECR_REPO = $(ECR_URL)/kubeflow-pipeline-test

login:
	aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $(ECR_URL)/

docker-base:
	docker buildx build -f src/base.Dockerfile \
	--no-cache \
	--platform linux/amd64 \
	--push -t $(ECR_REPO):brew_1146_base .

docker-src:
	docker buildx build -f src/Dockerfile \
	--build-arg BASE_IMAGE=$(ECR_REPO):brew_1146_base \
	--no-cache \
	--platform linux/amd64 \
	--push -t $(ECR_REPO):brew_1146_src .


build-docker: login \
	docker-base \
	docker-src

format:
	isort src/
	black src/

