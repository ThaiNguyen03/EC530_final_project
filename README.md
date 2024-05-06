# EC530_final_project

## Description
This is a project based on Project 2 DIYML. The aim of this project is to create an API to automate creating datasets, training and testing predefined models using these datasets, and then running inference based on the trained models.

## Note
This project use the transformers library to do the actual training process of the machine learning models. Therefore, the project currently only supports the models supported by the transformers library.

## Instructions for unit tests
Since the unit tests for the DataUpload modules use mock requests, users need to have the test_user folder, available in this repository, before running the remaining unit tests. This folder simulates the files already being uploaded using the API. The test_user folder should be on the same directory level as the App.py and the IntegrationTestFull.py before running unit tests.

## Instructions for testing the front end
The front end assumes that a test user, with username 'thai' and password '123' already presented in the mongodb Database. Please make sure the test user is set up before testing the front-end
## API information
The API provides the following endpoints:
- /login: compares user credentials with database to log in user
- /upload_images: allows users to upload images to be used for training and testing. Images should include both training and testing category.
- /upload_label: allows users to upload labels for the images
- /export_to_parquet: export the dataset to a folder and creates a parquet file each for the training and testing data
- /upload_parameters: allows users to upload their own parameters for training
- /start_training: users select a predefined model to train using the exported dataset and the uploaded parameters
- /get_training_stats: retrieve training stats after training
- /publish_model: publish the model to a folder on the server directory for use in testing and inference
- /test: runs inference on the test dataset uploaded earlier 
- /inference/<string:user_id>/<string:project_id>: allows users to run inference using a unique endpoint, will return a prediction onto the web interface
- /upload_inference: allows the user to upload a new image and will pass the path of the image on the server directory to the /inference/<string:user_id>/<string:project_id> endpoint