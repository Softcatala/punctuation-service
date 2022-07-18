build-all: docker-build

docker-build:
	docker build -t punctuation-service . -f docker/dockerfile

docker-run:
	docker run -it --rm -p 8000:8000 punctuation-service

