DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PROJECT_NAME=tdml-example
PROJECT_DIR=${PROJECT_NAME}

build:
	docker build --rm \
				 -f ${DIR}/Dockerfile \
				 -t ${PROJECT_NAME} \
				 ${DIR}

stop:
	docker rm -f ${PROJECT_NAME} || true

run_no_tests_titanic_pipeline: stop
	docker run \
			--rm \
			--name ${PROJECT_NAME} \
			-t \
			-v ${DIR}:/${PROJECT_DIR} \
			-e PROJECT_DIR=/tdml-example \
			-w /${PROJECT_DIR}/ \
			${PROJECT_NAME} \
			python3 -m no_tests.no_tests_titanic_ml_pipeline.py

