# TDML Example
#### Test Driven Machine Learning Example

#### Setup 
1) Requirements:
    * Docker
    * Make
    * Lots of bravery

2) Run the testable_titanic_ml_pipeline:

`make run_with_tests_titanic_pipeline`

3) Run the no_tests_titanic_ml_pipeline:

`make run_no_tests_titanic_pipeline`

#### Your Task

1) We received urgent new passenger data that we need to score to predict their survival:
    * `datasets/titanic/2_unknown_new_data_to_score.csv`
    
2) Unfortunately both pipelines are breaking when fed the new dataset.

3) Your task is to try to modify the `no_tests_titanic_ml_pipeline.py` code to support the new data and potentially save lives. Challenge yourself and set a 30 minute to get the no_tests code ready.

4) Once you give up on the no_tests challenge because of all the inter-dependencies in the code and the lack of tests, try to do the same on the `testable_titanic_ml_pipeline.py` code.

5) Send your solution in a pull request if you wish.  



Companion exercise to the "Clean Machine Learning Code" book: http://cleanmachinelearningcode.com/       


