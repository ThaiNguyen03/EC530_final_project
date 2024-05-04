import pytest
import logging
import tracemalloc
from unittest.mock import patch, MagicMock
from flask import Flask
from App import app, ImageUpload, LabelUpload, ParquetExport
import os
from App import app, PublishModel
from App import app, StartTraining, task_queue, task_complete_event
from App import app, GetTrainingStats, stats_collection
from datasets import load_from_disk, load_dataset
logging.basicConfig(filename='./testDataUpload.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testDataUpload.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)
tracemalloc.start()


def test_image_upload():
    with app.test_request_context():
        with patch('flask.request') as mock_request:
            mock_file = MagicMock()
            mock_file.filename = 'test_image.jpg'
            mock_request.files = {'file': mock_file}
            mock_request.form = {'user_id': 'test_user', 'project_id': 'test_project', 'image_id': 'test_image'}

            try:
                response = app.test_client().post('/upload_images', data = mock_request.form)
                assert response.status_code == 200
                assert b'Image uploaded successfully' in response.data
                mylogger.info('Image upload test passed successfully.')
            except Exception as e:
                mylogger.error(str(e))

    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()


def test_label_upload():
    with app.test_request_context():
        with patch('flask.request') as mock_request:
            label = 'test'
            mock_request.form = {'user_id': 'test_user',
                                 'project_id': 'test_project',
                                 'image_id': 'test_image',
                                 'label': label
                                 }

            try:
                response = app.test_client().post('/upload_label',data = mock_request.form)
                assert response.status_code == 200
                assert b'Label uploaded successfully' in response.data
                mylogger.info('Label upload test passed successfully.')
            except Exception as e:
                mylogger.error(str(e))

    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
def test_parquet_export():

    response = app.test_client().get('/export_to_parquet',json={
             'user_id':'test_user',
             'project_id':'test_project'
    })
    assert response.status_code == 200

    # with app.test_request_context():
    #     try:
    #         response = app.test_client().get('/export_to_parquet',json={
    #             'user_id':'test_user',
    #             'project_id':'test_project'
    #         })
    #         assert response.status_code == 200
    #         assert os.path.exists('image_data.parquet')
    #         mylogger.info('Parquet export test passed successfully.')
    #     except Exception as e:
    #         mylogger.error(str(e))
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
          #'train_dataset': './training_test/training_test.parquet'
          'train_dataset':'./DataUpload/user_dataset'
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



from App import app, InferenceAPI, task_queue_infer, task_complete_event_infer

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
def test_task_queue(client):
    test_set = load_dataset("food101", split="validation[:100]")
    # test_set.save_to_disk('./training_test')
    test_set.to_parquet('./inference.parquet')
    image = test_set["image"][10]
    image_path = "./inference_data"
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    image = image.save(f"{image_path}/image.png")
    image = test_set["image"][1]
    image_path = "./inference_data"
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    image = image.save(f"{image_path}/image1.png")
    with task_queue_infer.mutex:
        task_queue_infer.queue.clear()
    task_complete_event_infer.clear()
    rv1 = client.post('/inference/test_user/test_project', json={
        'model_name': 'test_model_path',
        'image_path': f'{image_path}/image.png',
        'model_path': '../Training/test_user/test_project/model'
    })
    rv2 = client.post('/inference/test_user/test_project', json={
        'model_name': 'test_model_path',
        'image_path': f'{image_path}/image1.png',
        'model_path': '../Training/test_user/test_project/model'
    })
    rv3 = client.post('/inference/test_user/test_project', json={
            'model_name': 'test_model_path',
            'image_path': f'{image_path}/image_wrong.png',
            'model_path': '../Training/test_user/test_project/model'
        })

    task_complete_event_infer.wait()
    assert task_queue_infer.empty()

    assert rv1.status_code == 200

    assert rv2.status_code == 200
    assert rv3.status_code == 404

#test dataset module test
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_module_api(client):
    test_set = load_dataset("food101", split="validation[:100]")
    test_set.to_parquet('./test.parquet')
    response = client.post('/test', json={
        'model_name': 'test_model_path',
        'dataset_path': './test.parquet',
        'user_id': 'test_user',
        'project_id':'test_project',
        #'model_path': '../Training/test_user/test_project/model'
    })
    assert response.status_code == 200
    assert 'results' in response.get_json()
    mylogger.info('Test for successful test_module passed.')





@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_testmodule_wrong_dataset(client):
    test_set = load_dataset("food101", split="validation[:100]")
    test_set.to_parquet('./test.parquet')
    response = client.post('/test', json={
        'model_name': 'test_model_path',
        'user_id':'test_user',
        'project_id':'test_project',
        'dataset_path': './test1.parquet',
       # 'model_path': '../Training/test_user/test_project/model'
    })

    assert response.status_code == 404
    mylogger.info('Test for check dataset passed.')


if __name__ == "__main__":
    pytest.main()
