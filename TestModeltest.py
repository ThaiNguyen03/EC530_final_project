import pytest
import logging
import tracemalloc
import PIL
from PIL import Image
from TestModel import app, InferenceAPI, task_queue, task_complete_event
import datasets
from datasets import load_dataset
import os

logging.basicConfig(filename='./testInference.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testInference.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)

# Start tracing memory allocations
tracemalloc.start()

#inference test
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client



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
    with task_queue.mutex:
        task_queue.queue.clear()
    task_complete_event.clear()
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

    task_complete_event.wait()
    assert task_queue.empty()

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
        'model_path': '../Training/test_user/test_project/model'
    })
    assert response.status_code == 404


    mylogger.info('Test for check dataset passed.')


# Stop tracing memory allocations
tracemalloc.stop()

current, peak = tracemalloc.get_traced_memory()
mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
