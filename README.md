## Ad click prediction using behavioral data

This project focusses on predicting whether a user will click on Ad or not based on certain inputs such as:

- Daily Time Spent on Site	
- Age
- Area Income
- Daily Internet Usage	
- Ad Topic Line	
- City
- Male
- Country

#### To install all the dependancies, run the following command
pip install -r requirements.txt

#### To run the model
- In bash: python main.py
- Then in the browser: http://<your-ip-address>:8080/docs
- Use the predict endpoint to predict whether the gievn user will click the ad or not 

### Project WorkFlow

1. Created a pipeline which involves the preprocessing steps such as encoding the categorical columns to numerical and scaling the values using MinMaxScaler.

2. Used various Machine Learning Algorithms such as Random Forest, Gradient Boosting, Support Vector Classifier and used GridSearchCV to get the optimal hyperparameters for the model.

3. Used accuracy metrics such as precision, recall, confusion matrix, roc-auc curves to determine which model to use. Additionally calculated and plotted the feature importances.

4. Saved the model using pickle and stored it in /models/final.pkl

5. With the help of Pydantic, FastAPI and Uvicorn, built the app which when give the various input features gives us a binary output(0/1). 
0 -- user will not click on the ad
1 -- user will click on the ad.


### Directory Structure

AD_CLICK_PREDICTION/
│
├── data/
│   └── advertising.csv    # Dataset for ad click prediction
│
├── env/                   # Virtual environment files 
│
├── experiments/           # Jupyter notebooks for experimentation
│   ├── eda.ipynb          # Exploratory Data Analysis notebook
│   ├── model.ipynb        # Model training and evaluation notebook
│   └── tests.ipynb        # Testing and debugging notebook
│
├── models/
│   ├── final.pkl          # Final saved model 
│   └── requirements.txt   # Dependencies specific to models
│
├── .dockerignore          # File to exclude files/folders in Docker builds
├── .env                   # Environment variables
├── .gitignore             # Git ignore file for untracked files
├── api.py                 # Script to handle API-related functionality
├── LICENSE                # Project license
├── main.py                # Main entry point for the project
├── README.md              # Project documentation
└── requirements.txt       # Main project dependencies
