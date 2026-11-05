import { Outlet } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import { getSystemStatus } from '@/services/api'
import type { SystemStatus } from '@/types'

interface Props {
  dataset: string
  onDatasetChange: (d: string) => void
}

export default function AppLayout({ dataset, onDatasetChange }: Props) {
  const [status, setStatus] = useState<SystemStatus | null>(null)

  useEffect(() => {
    getSystemStatus()
      .then(r => setStatus(r.data))
      .catch(() => setStatus(null))
    const t = setInterval(() => {
      getSystemStatus().then(r => setStatus(r.data)).catch(() => {})
    }, 30000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="flex h-screen bg-aero overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar
          dataset={dataset}
          onDatasetChange={onDatasetChange}
          available={status?.datasets_available ?? ['FD001']}
          modelsReady={status?.models_ready ?? false}
          backendOk={!!status}
        />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
