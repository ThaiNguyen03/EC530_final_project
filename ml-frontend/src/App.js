import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Login from './Login';
import DataUpload from './DataUpload';
import Parameters from './Parameters'; // Import Parameters component
import Training from './Training'
import Inference from './Inference'
function App() {
  const [userLoggedIn, setUserLoggedIn] = useState(false);
  const [userId, setUserId] = useState(null);

  const handleLogin = (userId) => {
    setUserId(userId);
    setUserLoggedIn(true);
  };

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={userLoggedIn ? <DataUpload user_id={userId} /> : <Login onLogin={handleLogin} />} />
          <Route path="/parameters" element={<Parameters user_id={userId} />} />
          <Route path="/training" element={<Training user_id={userId} />} />
          <Route path="/inference/:user_id" element={<Inference user_id={userId} />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
