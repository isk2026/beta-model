import React from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import BetaDashboard from './BetaDashboard.jsx'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BetaDashboard />
  </React.StrictMode>,
)
