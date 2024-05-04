// App.js

import React, { useState } from 'react';
import Login from './Login';
import DataUpload from './DataUpload';

function App() {
  const [userLoggedIn, setUserLoggedIn] = useState(false);
  const [userId, setUserId] = useState(null);

  const handleLogin = (userId) => {
    setUserId(userId);
    setUserLoggedIn(true);
  };

  return (
    <div className="App">
      {userLoggedIn ? (
        <DataUpload user_id={userId} />
      ) : (
        <Login onLogin={handleLogin} />
      )}
    </div>
  );
}

export default App;
