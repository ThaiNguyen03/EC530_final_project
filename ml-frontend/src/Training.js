import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
function Training({ user_id }) {
  const [project_id, setProjectId] = useState('');
  const [model_id, setModelId] = useState('');
  const [model_name, setModelName] = useState('');
  const [message, setMessage] = useState('');
  const [trainingStats, setTrainingStats] = useState(null);
  const [isPublished, setIsPublished] = useState(false);
  const [testResults, setTestResults] = useState('');

  const startTraining = async () => {
    try {
      const data = {
        user_id,
        project_id,
        model_name,
        model_id
      };
      const response = await axios.post('http://127.0.0.1:5000/start_training', data);
      if (response.status === 200) {
        setMessage(response.data.message);
        fetchTrainingStats();
      } else {
        setMessage(response.data.message);
      }
    } catch (error) {
      setMessage('Error starting training: ' + error.message);
    }
  };

  const fetchTrainingStats = async () => {
    try {
      const data = {
        user_id,
        project_id,
        model_name
      };
      const response = await axios.post('http://127.0.0.1:5000/get_training_stats', data);
      if (response.status === 200) {
        setTrainingStats(response.data);
      } else {
        setMessage('Error fetching training stats');
      }
    } catch (error) {
      setMessage('Error fetching training stats: ' + error.message);
    }
  };

  const publishModel = async () => {
    try {
      const data = {
        user_id,
        project_id,
        model_id
      };
      const response = await axios.post('http://127.0.0.1:5000/publish_model', data);
      if (response.status === 200) {
        setIsPublished(true);
        setMessage('Model published successfully');
      } else {
        setMessage('Error publishing model');
      }
    } catch (error) {
      setMessage('Error publishing model: ' + error.message);
    }
  };

  const testModel = async () => {
    try {
      const data = {
        user_id,
        project_id
      };
      const response = await axios.post('http://127.0.0.1:5000/test', data);
      if (response.status === 200) {
        setTestResults(response.data.results);
        setMessage(response.data.message);
      } else {
        setMessage(response.data.message);
      }
    } catch (error) {
      setMessage('Error testing model: ' + error.message);
    }
  };

  return (
    <div>
      <h2>Training</h2>
      <div>User ID: {user_id}</div>

      <input
        type="text"
        placeholder="Project ID"
        value={project_id}
        onChange={(e) => setProjectId(e.target.value)}
      />
      <input
        type="text"
        placeholder="Model ID"
        value={model_id}
        onChange={(e) => setModelId(e.target.value)}
      />
      <input
        type="text"
        placeholder="Model Name"
        value={model_name}
        onChange={(e) => setModelName(e.target.value)}
      />

      <button onClick={startTraining}>Start Training</button>
      {message && <div>{message}</div>}

      {trainingStats && (
        <div>
          <h3>Training Stats</h3>
          <input
            type="text"
            placeholder="Model ID"
            value={model_id}
            onChange={(e) => setModelId(e.target.value)}
          />
          {<button onClick={publishModel}>Publish</button>}
          <pre>{JSON.stringify(trainingStats, null, 2)}</pre>

          {<button onClick={testModel}>Test</button>}
          {testResults && <div>Test Results: {testResults}</div>}
          <Link to={`/inference/${user_id}`}>
            <button>Inference</button>
          </Link>
        </div>
      )}
    </div>
  );
}

export default Training;
