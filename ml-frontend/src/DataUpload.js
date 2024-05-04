// DataUpload.js
import React, { useState } from 'react';
import axios from 'axios';

function DataUpload({ user_id }) {
  const history = useHistory();
  const [project_id, setProjectId] = useState('');
  const [dataset_type, setDatasetType] = useState('train');
  const [label, setLabel] = useState('unknown');
  const [image_id, setImageId] = useState('');
  const [file, setFile] = useState(null);
  const handleStartTraining = () => {
    history.push(`/training/${user_id}`);
  };

  const handleFileUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', user_id);
    formData.append('project_id', project_id);
    formData.append('dataset_type', dataset_type);
    formData.append('label', label);
    formData.append('image_id', image_id);

    try {
      await axios.post('http://127.0.0.1:5000/upload_images', formData);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  return (
    <div>
      <div>User ID: {user_id}</div>
      <input type="text" placeholder="Project ID" value={project_id} onChange={(e) => setProjectId(e.target.value)} />
      <select value={dataset_type} onChange={(e) => setDatasetType(e.target.value)}>
        <option value="train">Train</option>
        <option value="test">Test</option>
      </select>
      <input type="text" placeholder="Label" value={label} onChange={(e) => setLabel(e.target.value)} />
      <input type="text" placeholder="Image ID" value={image_id} onChange={(e) => setImageId(e.target.value)} />
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleFileUpload}>Upload</button>
       <button onClick={handleStartTraining}>Start Training</button>
    </div>
  );
}

export default DataUpload;
