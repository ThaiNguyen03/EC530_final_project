from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient
import numpy as np
import sklearn
import evaluate
import torch
import PIL
import traceback
from datasets import load_from_disk, load_dataset
import json, os
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import Trainer, DefaultDataCollator, TrainingArguments, AutoModelForImageClassification, \
    AutoTokenizer, \
    AutoFeatureExtractor, AutoImageProcessor
from queue import Queue
from threading import Thread, Event
import shutil
task_complete_event = Event()
app = Flask(__name__)
api = Api(app)

# Initialize MongoDB client
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
model_collection = db['model_data']  # Collection to store model parameters
stats_collection = db['stats']  # Collection to store training stats
publish_model = db['publish_model']
task_queue = Queue()
results_dict = {}
def validate_parameters(params):
    errors = []
    if not isinstance(params['learning_rate'], (float, int)) or params['learning_rate'] <= 0:
        errors.append("learning_rate must be a positive number.")
    if not isinstance(params['per_device_train_batch_size'], int) or params['per_device_train_batch_size'] <= 0:
        errors.append("per_device_train_batch_size must be a positive integer.")
    if not isinstance(params['gradient_accumulation_steps'], int) or params['gradient_accumulation_steps'] <= 0:
        errors.append("gradient_accumulation_steps must be a positive integer.")
    if not isinstance(params['per_device_eval_batch_size'], int) or params['per_device_eval_batch_size'] <= 0:
        errors.append("per_device_eval_batch_size must be a positive integer.")
    if not isinstance(params['num_train_epochs'], int) or params['num_train_epochs'] <= 0:
        errors.append("num_train_epochs must be a positive integer.")
    if not isinstance(params['warmup_ratio'], (float,int)) or params['warmup_ratio'] < 0 or params['warmup_ratio'] > 1:
        errors.append("warmup_ratio must be a positive float between 0 and 1.")
    if not isinstance(params['logging_steps'], int) or params['logging_steps'] <= 0:
        errors.append("logging_steps must be a positive integer.")

    return errors

class UploadParameters(Resource):
    def post(self):
        data = request.get_json()
        task_queue.put(data)
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        parameters = data.get('parameters')
        validation_errors = validate_parameters(parameters)
        if validation_errors:
            return {"message": "Validation failed", "errors": validation_errors}, 400
        model_collection.insert_one({
            'user_id': user_id,
            'project_id': project_id,
            'parameters': parameters
        })
        return {"message": "Parameters uploaded successfully"}, 200





def start_training(data):
    try:
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        model_name = data.get('model_name')
        train_dataset_path = data.get('train_dataset')
        model_id = data.get('model_id')
        # Check if dataset exists
        if not os.path.exists(train_dataset_path):
            return {"message": f"Dataset {train_dataset_path} not found"}, 400

        parameters_data = model_collection.find_one({"user_id": user_id, "project_id": project_id})
        if not parameters_data:
            return {"message": "Parameters not found"}, 400

        parameters = parameters_data['parameters']
        api_dataset = load_dataset('parquet', data_files=train_dataset_path)

        api_dataset = api_dataset["train"].train_test_split(test_size=0.2)
        labels = api_dataset["train"].features["label"].names

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        checkpoint = model_name
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

        def transforms(examples):
            examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
            del examples["image"]
            return examples

        api_dataset = api_dataset.with_transform(transforms)
        model = AutoModelForImageClassification.from_pretrained(model_name,
                                                                num_labels=len(labels),
                                                                id2label=id2label,
                                                                label2id=label2id
                                                                )
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        model_saved_path = f'./{user_id}/{project_id}/model'
        training_args = TrainingArguments(
            output_dir=f'./results/{user_id}/{project_id}',
            num_train_epochs=parameters.get('num_train_epochs'),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=parameters.get('learning_rate'),
            per_device_train_batch_size=parameters.get('per_device_train_batch_size'),
            per_device_eval_batch_size=parameters.get('per_device_eval_batch_size'),
            gradient_accumulation_steps=parameters.get('gradient_accumulation_steps'),
            warmup_ratio=parameters.get('warmup_ratio'),
            logging_steps=parameters.get('logging_steps'),
            load_best_model_at_end=True,
            logging_dir=f'./logs/{user_id}/{project_id}',
            remove_unused_columns=False,
        )
        data_collator = DefaultDataCollator()
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=api_dataset["train"],
            eval_dataset=api_dataset["test"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(model_saved_path)
        absolute_model_path = os.path.abspath(model_saved_path)
        stats_collection.insert_one({
            'user_id': user_id,
            'project_id': project_id,
            'model_name': model_name,
            'model_id':model_id,
            'training_stats': trainer.evaluate(),
            'model_saved_path': absolute_model_path
        })
        return {"message": f"Training for model {model_name} completed successfully"}, 200

    except Exception as e:
        traceback.print_exc()
        return {"message": "Training failed", "error": str(e)}, 400


def worker(request_id):
    while not task_queue.empty():
        data = task_queue.get()
       # start_training(data)
        result, status_code = start_training(data)
       # print(result)
        results_dict[request_id] = (result, status_code)
        task_queue.task_done()
    task_complete_event.set()


class StartTraining(Resource):
    def post(self):
        data = request.get_json()
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        request_id = f"{user_id}_{project_id}"
        task_queue.put(data)
        task_complete_event.clear()
        worker_thread = Thread(target=worker,args=(request_id,))
        worker_thread.start()
        worker_thread.join()
        #task_complete_event.wait()
        if request_id in results_dict:
            result, status_code = results_dict.pop(request_id)
            return result, status_code
        else:
            return {"message": "Error occurred during inference"}, 500
        #return {"message": "Training request received"}, 200

class GetTrainingStats(Resource):
    def get(self):
        data = request.get_json()
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        model_name = data.get('model_name')
        stats = stats_collection.find_one({
            'user_id': user_id,
            'project_id': project_id,
            'model_name': model_name
        })

        if stats and 'training_stats' in stats:
            return stats['training_stats'], 200
        else:
            return {"message": "No training stats found"}, 404
class PublishModel(Resource):
    def post(self):
        data = request.get_json()
        model_id = data.get('model_id')
        user_id = data.get('user_id')
        project_id = data.get('project_id')

        # Search for the model in stats_collection
        model_entry = stats_collection.find_one({"model_id": model_id})
        if not model_entry:
            return {"message": f"Model with ID {model_id} not found"}, 404

        # Create a new folder for the published model if it doesn't exist
        publish_folder = f"./published_models/{user_id}/{project_id}"
        if os.path.exists(publish_folder):
            shutil.rmtree(publish_folder)

        # Copy the content from model_saved_path to the publish folder
        model_saved_path = model_entry['model_saved_path']
        shutil.copytree(model_saved_path, publish_folder)


        try:
            publish_model.insert_one({
                'user_id': user_id,
                'project_id': project_id,
                'model_id': model_id,
                'published_folder_path': os.path.abspath(publish_folder)
            })
        except Exception as e:
            return {"message": "Model data not published"}, 404

        return {"message": f"Model {model_id} published successfully"}, 200

api.add_resource(UploadParameters, '/upload_parameters')
api.add_resource(StartTraining, '/start_training')
api.add_resource(GetTrainingStats, '/get_training_stats')
api.add_resource(PublishModel, '/publish_model')
if __name__ == '__main__':
    app.run(debug=True)
