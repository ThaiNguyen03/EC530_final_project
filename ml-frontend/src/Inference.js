import React, { useState } from 'react';
import axios from 'axios';

function Inference({ user_id }) {
  const [project_id, setProjectId] = useState('');
  const [image, setImage] = useState(null);
  const [imagePath, setImagePath] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleFileUpload = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append('file', image);
    formData.append('user_id', user_id);
    formData.append('project_id', project_id);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload_inference', formData);

      setImagePath(response.data.file_path);


      const inferenceData = {

        image_path: imagePath
      };


      const inferenceResponse = await axios.post(`http://127.0.0.1:5000/inference/${user_id}/${project_id}`, inferenceData);


      setPrediction(inferenceResponse.data.results);
    } catch (error) {
      console.error('Inference failed:', error);
    }
  };

  return (
    <div>
      <h2>Inference</h2>
      <div>User ID: {user_id}</div>
      <input
        type="text"
        placeholder="Project ID"
        value={project_id}
        onChange={(e) => setProjectId(e.target.value)}
      />
      <input type="file" onChange={(e) => setImage(e.target.files[0])} />
      <button onClick={handleFileUpload}>Upload</button>
      {prediction && <div>Prediction: {prediction}</div>}
    </div>
  );
}

export default Inference;
