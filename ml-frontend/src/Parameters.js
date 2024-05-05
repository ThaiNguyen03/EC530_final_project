import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
function Parameters({ user_id }) {
  const [project_id, setProjectId] = useState('');
  const [learningRate, setLearningRate] = useState('');
  const [perDeviceTrainBatchSize, setPerDeviceTrainBatchSize] = useState('');
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState('');
  const [perDeviceEvalBatchSize, setPerDeviceEvalBatchSize] = useState('');
  const [numTrainEpochs, setNumTrainEpochs] = useState('');
  const [warmupRatio, setWarmupRatio] = useState('');
  const [loggingSteps, setLoggingSteps] = useState('');
  const [message, setMessage] = useState('');

  const uploadParameters = async () => {
    try {
      const formData = new FormData();
      formData.append('user_id', user_id);
      formData.append('project_id', project_id);
      formData.append('learning_rate', learningRate);
      formData.append('per_device_train_batch_size', perDeviceTrainBatchSize);
      formData.append('gradient_accumulation_steps', gradientAccumulationSteps);
      formData.append('per_device_eval_batch_size', perDeviceEvalBatchSize);
      formData.append('num_train_epochs', numTrainEpochs);
      formData.append('warmup_ratio', warmupRatio);
      formData.append('logging_steps', loggingSteps);

      const response = await axios.post('http://127.0.0.1:5000/upload_parameters', formData);
      if (response.status === 200) {
        setMessage(response.data.message);
      } else {
        setMessage(response.data);
      }
    } catch (error) {
      setMessage('Error uploading parameters: ' + error.message);
    }
  };

  return (
    <div>
      <h2>Parameters</h2>
      <div>User ID: {user_id}</div>
      <input
        type="text"
        placeholder="Project ID"
        value={project_id}
        onChange={(e) => setProjectId(e.target.value)}
      />

      <div>
        <input
          type="number"
          placeholder="Learning Rate"
          value={learningRate}
          onChange={(e) => setLearningRate(e.target.value)}
        />
      </div>
      <div>
        <input
          type="number"
          placeholder="Per Device Train Batch Size"
          value={perDeviceTrainBatchSize}
          onChange={(e) => setPerDeviceTrainBatchSize(e.target.value)}
        />
      </div>
      <div>
        <input
          type="number"
          placeholder="Gradient Accumulation Steps"
          value={gradientAccumulationSteps}
          onChange={(e) => setGradientAccumulationSteps(e.target.value)}
        />
      </div>
      <div>
        <input
          type="number"
          placeholder="Per Device Eval Batch Size"
          value={perDeviceEvalBatchSize}
          onChange={(e) => setPerDeviceEvalBatchSize(e.target.value)}
        />
      </div>
      <div>
        <input
          type="number"
          placeholder="Number of Train Epochs"
          value={numTrainEpochs}
          onChange={(e) => setNumTrainEpochs(e.target.value)}
        />
      </div>
      <div>
        <input
          type="number"
          placeholder="Warmup Ratio"
          value={warmupRatio}
          onChange={(e) => setWarmupRatio(e.target.value)}
        />
      </div>
      <div>
        <input
          type="number"
          placeholder="Logging Steps"
          value={loggingSteps}
          onChange={(e) => setLoggingSteps(e.target.value)}
        />
      </div>
      <button onClick={uploadParameters}>Upload Parameters</button>
      <Link to="/training">
        <button>Go to Training</button>
      </Link>
      {message && <div>{message}</div>}
    </div>
  );
}

export default Parameters;
