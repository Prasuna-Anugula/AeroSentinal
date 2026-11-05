import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ToastProvider } from '@/components/ui/Toast'
import AppLayout from '@/components/layout/AppLayout'

import Landing           from '@/pages/Landing'
import Overview          from '@/pages/Overview'
import EngineMonitor     from '@/pages/EngineMonitor'
import RULPrediction     from '@/pages/RULPrediction'
import FailureRisk       from '@/pages/FailureRisk'
import AnomalyDetection  from '@/pages/AnomalyDetection'
import SensorIntelligence from '@/pages/SensorIntelligence'
import MaintenancePlanner from '@/pages/MaintenancePlanner'
import DatasetExplorer   from '@/pages/DatasetExplorer'
import ModelInsights     from '@/pages/ModelInsights'
import Settings          from '@/pages/Settings'

export default function App() {
  const [dataset, setDataset] = useState('FD001')

  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<AppLayout dataset={dataset} onDatasetChange={setDataset} />}>
            <Route index element={<Navigate to="overview" replace />} />
            <Route path="overview"    element={<Overview dataset={dataset} />} />
            <Route path="engines"     element={<EngineMonitor dataset={dataset} />} />
            <Route path="rul"         element={<RULPrediction dataset={dataset} />} />
            <Route path="failure"     element={<FailureRisk dataset={dataset} />} />
            <Route path="anomaly"     element={<AnomalyDetection dataset={dataset} />} />
            <Route path="sensors"     element={<SensorIntelligence dataset={dataset} />} />
            <Route path="maintenance" element={<MaintenancePlanner dataset={dataset} />} />
            <Route path="datasets"    element={<DatasetExplorer dataset={dataset} />} />
            <Route path="models"      element={<ModelInsights />} />
            <Route path="settings"    element={<Settings />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  )
}
