import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { Search } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import StatusBadge from '@/components/ui/StatusBadge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { getFleet, getEngineSnapshot, getEngineSensors } from '@/services/api'
import type { FleetEngine, SensorSnapshot, SensorHistory } from '@/types'
import { formatRUL, formatProb, healthColor } from '@/utils'

const SENSORS_TO_SHOW = ['sensor2','sensor3','sensor4','sensor7','sensor8','sensor11','sensor12','sensor13','sensor15','sensor17','sensor20','sensor21']

interface Props { dataset: string }

const TT = { contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 11 }, labelStyle: { color: '#9AAAC2' } }
const COLORS = ['#69C9FF','#8B83FF','#36E0C1','#FFBF69','#39D98A','#FF647C','#FB923C','#A78BFA','#34D399','#F472B6','#60A5FA','#FCD34D']

export default function EngineMonitor({ dataset }: Props) {
  const [fleet,    setFleet]    = useState<FleetEngine[]>([])
  const [search,   setSearch]   = useState('')
  const [selected, setSelected] = useState<FleetEngine | null>(null)
  const [snapshot, setSnapshot] = useState<SensorSnapshot[]>([])
  const [history,  setHistory]  = useState<SensorHistory[]>([])
  const [activeSensors, setActiveSensors] = useState<string[]>(['sensor2','sensor7','sensor11'])
  const [loading,  setLoading]  = useState(true)

  useEffect(() => {
    setLoading(true)
    getFleet(dataset).then(r => {
      setFleet(r.data)
      setSelected(r.data[0] ?? null)
    }).finally(() => setLoading(false))
  }, [dataset])

  useEffect(() => {
    if (!selected) return
    getEngineSnapshot(dataset, selected.engine_unit).then(r => setSnapshot(r.data.sensors ?? []))
    getEngineSensors(dataset, selected.engine_unit).then(r => setHistory(r.data.sensors ?? []))
  }, [selected, dataset])

  const filtered = fleet.filter(e =>
    String(e.engine_unit).includes(search) ||
    e.health_status.includes(search.toLowerCase())
  )

  // Build chart data from selected sensors
  const chartData = (() => {
    if (!history.length) return []
    const base = history[0]
    return base.cycles.map((cycle, i) => {
      const point: Record<string, number> = { cycle }
      activeSensors.forEach(s => {
        const h = history.find(h => h.sensor === s)
        if (h) point[s] = h.values[i]
      })
      return point
    })
  })()

  if (loading) return <PageLoader />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display font-bold">Engine Monitor</h1>

      <div className="grid lg:grid-cols-3 gap-5">
        {/* Fleet list */}
        <GlassCard className="lg:col-span-1 !p-0 overflow-hidden">
          <div className="p-4 border-b border-border">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-sec" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search engine…"
                className="w-full bg-white/5 border border-border rounded-lg pl-8 pr-3 py-2 text-sm text-text-pri placeholder-text-sec/40 focus:outline-none focus:border-primary/40"
              />
            </div>
          </div>
          <div className="overflow-y-auto max-h-[60vh]">
            {filtered.map(e => (
              <button
                key={e.engine_unit}
                onClick={() => setSelected(e)}
                className={`w-full flex items-center justify-between px-4 py-3 border-b border-border/50 transition-colors text-left ${selected?.engine_unit === e.engine_unit ? 'bg-primary/10' : 'hover:bg-white/3'}`}
              >
                <div>
                  <p className="font-mono text-sm font-medium text-text-pri">ENG-{String(e.engine_unit).padStart(3,'0')}</p>
                  <p className="text-xs text-text-sec">Cycle {e.last_cycle}</p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-sm font-bold" style={{ color: healthColor(e.health_status) }}>
                    {formatRUL(e.predicted_rul)}
                  </p>
                  <StatusBadge value={e.health_status} className="text-[10px]" />
                </div>
              </button>
            ))}
          </div>
        </GlassCard>

        {/* Engine detail */}
        <div className="lg:col-span-2 space-y-4">
          {selected && (
            <>
              {/* Header */}
              <GlassCard className="!p-4">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div>
                    <h2 className="font-display font-bold text-xl">Engine {selected.engine_unit}</h2>
                    <p className="text-text-sec text-sm font-mono">{dataset} · Last Cycle: {selected.last_cycle}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <StatusBadge value={selected.health_status} />
                    {selected.risk_level && <StatusBadge value={selected.risk_level} label={`Risk: ${selected.risk_level}`} />}
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
                  <div className="glass-sm p-3">
                    <p className="text-xs text-text-sec font-mono">Predicted RUL</p>
                    <p className="text-lg font-bold text-primary">{formatRUL(selected.predicted_rul)}</p>
                  </div>
                  <div className="glass-sm p-3">
                    <p className="text-xs text-text-sec font-mono">True RUL</p>
                    <p className="text-lg font-bold text-text-pri">{selected.true_rul !== null ? formatRUL(selected.true_rul) : '—'}</p>
                  </div>
                  <div className="glass-sm p-3">
                    <p className="text-xs text-text-sec font-mono">Failure Prob</p>
                    <p className="text-lg font-bold text-warning">{formatProb(selected.failure_probability)}</p>
                  </div>
                  <div className="glass-sm p-3">
                    <p className="text-xs text-text-sec font-mono">Anomaly</p>
                    <p className="text-lg font-bold text-text-pri">{selected.anomaly_severity ?? '—'}</p>
                  </div>
                </div>
              </GlassCard>

              {/* Sensor selector */}
              <GlassCard>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide">Sensor Degradation Trends</h3>
                </div>
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {SENSORS_TO_SHOW.map((s, i) => (
                    <button
                      key={s}
                      onClick={() => setActiveSensors(p => p.includes(s) ? p.filter(x => x !== s) : [...p, s])}
                      className={`text-xs font-mono px-2.5 py-1 rounded-full border transition-colors ${activeSensors.includes(s) ? 'border-primary/50 bg-primary/10 text-primary' : 'border-border text-text-sec hover:border-primary/30'}`}
                      style={activeSensors.includes(s) ? { borderColor: COLORS[i % COLORS.length] + '80', color: COLORS[i % COLORS.length] } : {}}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
                    <XAxis dataKey="cycle" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#9AAAC2', fontSize: 10 }} width={45} />
                    <Tooltip {...TT} />
                    <Legend wrapperStyle={{ fontSize: 11, color: '#9AAAC2' }} />
                    {activeSensors.map((s, i) => (
                      <Line key={s} type="monotone" dataKey={s} stroke={COLORS[SENSORS_TO_SHOW.indexOf(s) % COLORS.length]} dot={false} strokeWidth={1.5} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </GlassCard>

              {/* Sensor snapshot table */}
              <GlassCard>
                <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-3">Sensor Snapshot</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-border">
                        {['Sensor','Latest','Mean','Min','Max'].map(h => (
                          <th key={h} className="text-left py-2 px-2 text-text-sec">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.map(s => (
                        <tr key={s.sensor} className="border-b border-border/30 hover:bg-white/2 transition-colors">
                          <td className="py-1.5 px-2 text-primary">{s.sensor}</td>
                          <td className="py-1.5 px-2 text-text-pri font-bold">{s.latest_value.toFixed(3)}</td>
                          <td className="py-1.5 px-2 text-text-sec">{s.mean_value.toFixed(3)}</td>
                          <td className="py-1.5 px-2 text-text-sec">{s.min_value.toFixed(3)}</td>
                          <td className="py-1.5 px-2 text-text-sec">{s.max_value.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
