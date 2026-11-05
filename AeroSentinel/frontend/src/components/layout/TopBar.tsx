import { useState, useEffect } from 'react'
import { Wifi, WifiOff, ChevronDown, Clock } from 'lucide-react'
import { cn } from '@/utils'

interface Props {
  dataset: string
  onDatasetChange: (d: string) => void
  available: string[]
  modelsReady: boolean
  backendOk: boolean
}

export default function TopBar({ dataset, onDatasetChange, available, modelsReady, backendOk }: Props) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <header className="h-14 bg-bg-sec/80 backdrop-blur border-b border-border flex items-center justify-between px-6 shrink-0">
      {/* Left */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-text-sec uppercase tracking-widest">AeroSentinel</span>
        <span className="text-text-sec/40 mx-1">//</span>
        <span className="text-xs font-mono text-primary">Mission Control</span>
      </div>

      {/* Right */}
      <div className="flex items-center gap-4">
        {/* Dataset selector */}
        <div className="relative">
          <select
            value={dataset}
            onChange={e => onDatasetChange(e.target.value)}
            className="appearance-none bg-white/5 border border-border text-text-pri text-xs font-mono rounded-lg px-3 py-1.5 pr-7 focus:outline-none focus:border-primary/40 cursor-pointer"
          >
            {available.map(d => (
              <option key={d} value={d} className="bg-bg-sec">{d}</option>
            ))}
          </select>
          <ChevronDown size={10} className="absolute right-2 top-1/2 -translate-y-1/2 text-text-sec pointer-events-none" />
        </div>

        {/* Model status */}
        <div className={cn(
          'flex items-center gap-1.5 text-xs font-mono px-2.5 py-1 rounded-full border',
          modelsReady ? 'text-success border-success/30 bg-success/10' : 'text-warning border-warning/30 bg-warning/10'
        )}>
          <span className={cn('w-1.5 h-1.5 rounded-full', modelsReady ? 'bg-success animate-pulse' : 'bg-warning')} />
          {modelsReady ? 'ML READY' : 'ML LOADING'}
        </div>

        {/* Backend */}
        <div className={cn(
          'flex items-center gap-1.5 text-xs font-mono',
          backendOk ? 'text-success' : 'text-danger'
        )}>
          {backendOk ? <Wifi size={12} /> : <WifiOff size={12} />}
          <span>{backendOk ? 'CONNECTED' : 'OFFLINE'}</span>
        </div>

        {/* Clock */}
        <div className="flex items-center gap-1.5 text-xs font-mono text-text-sec">
          <Clock size={11} />
          {time.toLocaleTimeString()}
        </div>
      </div>
    </header>
  )
}
