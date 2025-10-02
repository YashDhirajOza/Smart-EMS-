import React from 'react';
import EnergyFlowDiagram from './components/EnergyFlowDiagram';
import AssetHeartbeat from './components/AssetHeartbeat';
import AlertsPanel from './components/AlertsPanel';
import AdvisoryPanel from './components/AdvisoryPanel';
import './styles/app.css';

const App: React.FC = () => {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>UrjaNet Living Energy Flow</h1>
      </header>
      <main className="app-main">
        <section className="energy-flow">
          <EnergyFlowDiagram />
        </section>
        <section className="side-panels">
          <div className="panel">
            <h2>Component Heartbeats</h2>
            <AssetHeartbeat />
          </div>
          <div className="panel">
            <h2>Smart Alerts</h2>
            <AlertsPanel />
          </div>
          <div className="panel">
            <h2>Advisory</h2>
            <AdvisoryPanel />
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
