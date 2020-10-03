FROM python:3.7

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install pytest
RUN pip install scikit-learn


