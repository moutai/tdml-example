DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PROJECT_NAME=tdml-example
PROJECT_DIR=${PROJECT_NAME}

edit = edit
edit = edit

build_dev:
	docker build --rm \
				 -f ${DIR}/Dockerfile \
				 -t ${PROJECT_NAME} \
				 --target dev \
				 ${DIR}

build_prod:
	docker build --rm \
				 -f ${DIR}/Dockerfile \
				 -t ${PROJECT_NAME} \
				 --target prod \
				 ${DIR}

stop:
	docker rm -f ${PROJECT_NAME} || true

run_no_tests_titanic_pipeline: stop build_dev
	docker run \
			--rm \
			--name ${PROJECT_NAME} \
			-t \
			-v ${DIR}:/${PROJECT_DIR} \
			-e PROJECT_DIR=/tdml-example \
			-w /${PROJECT_DIR}/ \
			${PROJECT_NAME} \
			python3 -m no_tests.no_tests_titanic_ml_pipeline

run_with_tests_titanic_pipeline: stop build_dev
	docker run \
			--rm \
			--name ${PROJECT_NAME} \
			-t \
			-v ${DIR}:/${PROJECT_DIR} \
			-e PROJECT_DIR=/tdml-example \
			-w /${PROJECT_DIR}/ \
			${PROJECT_NAME} \
			pytest


run_no_tests_titanic_pipeline_with_new_data: stop build_dev
	docker run \
			--rm \
			--name ${PROJECT_NAME} \
			-t \
			-v ${DIR}:/${PROJECT_DIR} \
			-e PROJECT_DIR=/tdml-example \
			-e NEW_DATA=2_unknown_new_data_to_score.csv \
			-w /${PROJECT_DIR}/ \
			${PROJECT_NAME} \
			python3 -m no_tests.no_tests_titanic_ml_pipeline

run_with_tests_titanic_pipeline_with_new_data: stop build_dev
	docker run \
			--rm \
			--name ${PROJECT_NAME} \
			-t \
			-v ${DIR}:/${PROJECT_DIR} \
			-e PROJECT_DIR=/tdml-example \
			-e NEW_DATA=2_unknown_new_data_to_score.csv \
			-w /${PROJECT_DIR}/ \
			${PROJECT_NAME} \
			pytest