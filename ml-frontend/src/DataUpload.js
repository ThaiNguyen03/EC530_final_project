// DataUpload.js
import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
function DataUpload({ user_id }) {

  const [project_id, setProjectId] = useState('');
  const [dataset_type, setDatasetType] = useState('train');
  const [label, setLabel] = useState('unknown');
  const [image_id, setImageId] = useState('');
  const [file, setFile] = useState(null);


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
   const handleExportDataset = async () => {
    try {
      // Example of exporting dataset
      await axios.post('http://127.0.0.1:5000/export_to_parquet', {user_id,project_id });
    } catch (error) {
      console.error('Export dataset failed:', error);
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
      <button onClick={handleExportDataset}>Export Dataset</button>
      <Link to="/parameters">
        <button>Start Training</button>
      </Link>
    </div>
  );
}

export default DataUpload;