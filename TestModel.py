from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
from transformers import AutoModelForImageClassification, AutoTokenizer, AutoImageProcessor
from transformers import pipeline
import os
import pymongo
from pymongo import MongoClient
import torch
from queue import Queue
from datasets import load_dataset
from threading import Thread, Event
app = Flask(__name__)
api = Api(app)

task_queue = Queue()
results_dict = {}
task_complete_event = Event()
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
model_collection = db['model_data']  # Collection to store model parameters
publish_model = db['publish_model']
class TestAPI(Resource):
    def post(self):
        data = request.get_json()
        model_path = data.get('model_path')
        user_id= data.get('user_id')
        project_id = data.get('project_id')
        dataset_path = data.get('dataset_path')
        model_name = data.get('model_name')
        try:
            model_entry = publish_model.find_one({"user_id":user_id,"project_id":project_id})
            model_path = model_entry['published_folder_path']
        except Exception as e:
            return {"message":"published model not found"},404
        # Load the dataset
        try:
            api_dataset = load_dataset('parquet', data_files=dataset_path)
        except Exception as e:
            return {"message": "dataset not found"}, 404

        results = []

        for i in range(len(api_dataset["train"])):
            try:
                pil_image = api_dataset["train"][i]["image"]
            except Exception as e:
                return {"message": "Image not found"}, 404
            image_processor = AutoImageProcessor.from_pretrained(model_path)
            inputs = image_processor(pil_image, return_tensors="pt")

            my_model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
            with torch.no_grad():
                logits = my_model(**inputs).logits
            predicted = logits.argmax(-1).item()

            results.append(my_model.config.id2label[predicted])

        return {"message": "Inference run successfully",
                "results": results
                }, 200




def inference(data,user_id,project_id):
    #model_path = data.get('model_path')
    image_path = data.get('image_path')
    try:
        entry = publish_model.find_one({
        'user_id':user_id,
        'project_id':project_id
        })
        model_path = entry['published_folder_path']
    except Exception as e:
        return {"message": "published model not found"}, 404
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        return {"message": "Image not found"}, 404

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    inputs = image_processor(pil_image, return_tensors="pt")

    my_model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    with torch.no_grad():
        logits = my_model(**inputs).logits
    predicted = logits.argmax(-1).item()

    return {"message": "Inference run successfully", "results": my_model.config.id2label[predicted]}, 200


def worker(request_id,user_id,project_id):
    while not task_queue.empty():
        data = task_queue.get()
        result, status_code = inference(data,user_id,project_id)
        print(result)
        results_dict[request_id] = (result, status_code)

        task_queue.task_done()
    task_complete_event.set()


class InferenceAPI(Resource):
    def post(self, user_id, project_id):
        data = request.get_json()
        request_id = f"{user_id}_{project_id}"
        task_queue.put(data)
        task_complete_event.clear()
        worker_thread = Thread(target=worker, args=(request_id,user_id,project_id,))
        worker_thread.start()
        worker_thread.join()
        #task_complete_event.wait()
        if request_id in results_dict:
            result, status_code = results_dict.pop(request_id)
            return result, status_code
        else:
            return {"message": "Error occurred during inference"}, 500


api.add_resource(InferenceAPI, '/inference/<string:user_id>/<string:project_id>')
api.add_resource(TestAPI, '/test')

if __name__ == '__main__':
    app.run(debug=True)
