// Login.js

import React, { useState } from 'react';
import axios from 'axios';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [userId, setUserId] = useState(null); // State variable to hold user_id

  const handleLogin = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/login', { username, password });
      const { user_id } = response.data;
      onLogin(user_id); // Pass user_id to parent component
      setUserId(user_id); // Set user_id in state after successful login
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  return (
    <div>
      <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
      <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <button onClick={handleLogin}>Login</button>
      {userId && <p>User ID: {userId}</p>} {/* Display user_id if it exists */}
    </div>
  );
}

export default Login;
