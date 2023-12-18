# ML Based Smart Search

[This projects implements Logistic regression and SVC to provide the search with auto fill and suggestions for direct link to get hyperlinked to a specific part within the website]

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```



## Running the Smart Search using FastAPI Server
```bash
To run the FastAPI server, use the following command:
uvicorn auto_fill3_FastAPI:app --reload
```

## Training with Project Data
To train the model with the project data, run:
```bash
python svm_logReg_training.py
```
## Training with Custom Data
If you want to train the model with custom data, update the csv_data_path variable in svm_logReg_training.py to your .csv file location and then run:
```bash
python svm_logReg_training.py
```

## Local Inference
For local inference, use the following command:
```bash
python inference.py
```

## Using the Machine Learning Product in the Browser
To use the machine learning product in the browser, run the FastAPI server again:

```bash

uvicorn auto_fill3_FastAPI:app --reload
```




