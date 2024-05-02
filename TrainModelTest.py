import pytest
import logging
import tracemalloc

from datasets import load_dataset
import os
import pytest
from TrainModel import app, PublishModel
from TrainModel import app, StartTraining, task_queue, task_complete_event
from TrainModel import app, GetTrainingStats, stats_collection
logging.basicConfig(filename='./testTraining.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testTraining.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)
tracemalloc.start()


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_upload_parameters(client):
    test_set = load_dataset("food101", split="train[:100]")
    test_set.save_to_disk('./training_test')
    test_set.to_parquet('./training_test/training_test.parquet')
    rv = client.post('/upload_parameters', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'parameters': {
            'learning_rate': 5e-5,
            'per_device_train_batch_size': 6,
            'gradient_accumulation_steps': 4,
            'per_device_eval_batch_size': 6,
            'num_train_epochs': 3,
            'warmup_ratio': 0.1,
            'logging_steps': 10,
        },

    })
    assert b'Parameters uploaded successfully' in rv.data


def test_start_training(client):
      test_set = load_dataset("food101", split="train[:100]")
      test_set.save_to_disk('./training_test')
      test_set.to_parquet('./training_test/training_test.parquet')
      with task_queue.mutex:
          task_queue.queue.clear()
      task_complete_event.clear()
      rv = client.post('/start_training', json={
          'user_id': 'test_user',
          'project_id': 'test_project',
          'model_id':'model1',
          'model_name': 'google/vit-base-patch16-224-in21k',
          'train_dataset': './training_test/training_test.parquet'
      })
      try:
          task_complete_event.wait()
          assert task_queue.empty()
          assert rv.status_code == 200
          mylogger.info("test_start_training passed")
      except AssertionError:
          mylogger.error("test_start_training not passed")

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        stats_collection.insert_one({
            'user_id': 'test_user3',
            'project_id': 'test_project3',
            'model_name': 'google/vit-base-patch16-224-in21k',
            'training_stats': {
                'eval_loss': '0.2',
                'eval_error': '0.3',
                'accuracy': '0.92'
            },
            'model_save_path':'home_model'
        })
        yield client


def test_get_training_stats(client):
    rv = client.get('/get_training_stats', json={
        'user_id': 'test_user3',
        'project_id': 'test_project3',
        'model_name': 'google/vit-base-patch16-224-in21k'
    })
    try:
        assert rv.status_code == 200
        mylogger.info("Test passed")
    except AssertionError:
        mylogger.error("Test failed")


def test_wrong_training_stats(client):
    rv = client.get('/get_training_stats', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'wrong_model'
    })
    try:
        assert rv.status_code == 404
        mylogger.info("Test passed")
    except AssertionError:
        mylogger.error("Test failed")
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_publish_model_success(client):
    # Send a POST request to publish the model
    response = client.post('/publish_model', json={
        'model_id': 'model1',
        'user_id': 'test_user',
        'project_id': 'test_project'
    })
    try:
        assert response.status_code == 200
        mylogger.info("Test for successful publish passed")
    except AssertionError:
        mylogger.error("Test for successful publish not passed")





@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_publish_model_wrong_credentials(client):
    # Send a POST request to publish the model
    response = client.post('/publish_model', json={
        'model_id': 'model100',
        'user_id': 'test_user',
        'project_id': 'test_project'
    })
    try:
        assert response.status_code == 404
        mylogger.info("Test for wrong credentials passed")
    except AssertionError:
        mylogger.error("Test for wrong credentials not passed")
