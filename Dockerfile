FROM python:3.7 as dev

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install pytest
RUN pip install scikit-learn

FROM dev as prod
COPY . /tdml-example
CMD ["pytest", "/tdml-example/with_tests/tests"]




