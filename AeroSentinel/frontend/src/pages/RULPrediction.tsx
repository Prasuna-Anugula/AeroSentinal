import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Activity, RefreshCw } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import StatusBadge from '@/components/ui/StatusBadge'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { useToast } from '@/components/ui/Toast'
import { getEngines, predictRUL, getPredictionHistory } from '@/services/api'
import type { RULPrediction as RULResult, HealthStatus } from '@/types'
import { healthColor, formatRUL } from '@/utils'

interface Props { dataset: string }

const TT = { contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 11 } }

function RULGauge({ rul, max = 300 }: { rul: number; max?: number }) {
  const pct   = Math.min(rul / max, 1)
  const angle = -135 + pct * 270
  const status: HealthStatus = rul > 100 ? 'healthy' : rul > 50 ? 'warning' : rul > 20 ? 'high_risk' : 'critical'
  const color = healthColor(status)

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-44 h-44">
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <circle cx="100" cy="100" r="80" fill="none" stroke="rgba(148,180,220,0.1)" strokeWidth="14" strokeDasharray="251 502" strokeDashoffset="-125" strokeLinecap="round" />
          <motion.circle
            cx="100" cy="100" r="80"
            fill="none" stroke={color} strokeWidth="14"
            strokeDasharray={`${pct * 251} 502`} strokeDashoffset="-125"
            strokeLinecap="round"
            initial={{ strokeDasharray: '0 502' }}
            animate={{ strokeDasharray: `${pct * 251} 502` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
          {/* Needle */}
          <motion.line
            x1="100" y1="100"
            x2={100 + 65 * Math.cos((angle - 90) * Math.PI / 180)}
            y2={100 + 65 * Math.sin((angle - 90) * Math.PI / 180)}
            stroke={color} strokeWidth="2.5" strokeLinecap="round"
            initial={false}
            animate={{
              x2: 100 + 65 * Math.cos((angle - 90) * Math.PI / 180),
              y2: 100 + 65 * Math.sin((angle - 90) * Math.PI / 180),
            }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
          <circle cx="100" cy="100" r="5" fill={color} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center mt-6">
          <motion.span
            key={rul}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-3xl font-bold font-display"
            style={{ color }}
          >
            {Math.round(rul)}
          </motion.span>
          <span className="text-xs font-mono text-text-sec">cycles</span>
        </div>
      </div>
      <StatusBadge value={status} />
    </div>
  )
}

export default function RULPrediction({ dataset }: Props) {
  const { toast } = useToast()
  const [engines,   setEngines]   = useState<number[]>([])
  const [engine,    setEngine]    = useState(1)
  const [modelKey,  setModelKey]  = useState('rul_rf')
  const [result,    setResult]    = useState<RULResult | null>(null)
  const [history,   setHistory]   = useState<any[]>([])
  const [loading,   setLoading]   = useState(false)

  useEffect(() => {
    getEngines(dataset).then(r => {
      const units: number[] = r.data.engines ?? []
      setEngines(units)
      if (units.length) setEngine(units[0])
    })
    getPredictionHistory(20).then(r => setHistory(r.data))
  }, [dataset])

  async function run() {
    setLoading(true)
    try {
      const r = await predictRUL(dataset, engine, modelKey)
      setResult(r.data)
      getPredictionHistory(20).then(r => setHistory(r.data))
      toast('RUL prediction complete', 'success')
    } catch (e: any) {
      toast(e.response?.data?.detail ?? 'Prediction failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const histChartData = history
    .filter(h => h.dataset_subset === dataset && h.predicted_rul !== null)
    .slice(0, 15)
    .reverse()
    .map(h => ({ engine: `E${h.engine_unit}`, rul: Math.round(h.predicted_rul) }))

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display font-bold">RUL Prediction</h1>
      <p className="text-text-sec text-sm -mt-4">Remaining Useful Life estimation using Random Forest regression (FD001 MAE: 13.15 cycles, R²: 0.804)</p>

      <div className="grid lg:grid-cols-5 gap-5">
        {/* Controls */}
        <GlassCard className="lg:col-span-2 space-y-4">
          <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Prediction Parameters</h3>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Dataset</label>
            <div className="text-sm font-mono px-3 py-2 bg-white/5 rounded-lg border border-border text-primary">{dataset}</div>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Engine Unit</label>
            <select
              value={engine}
              onChange={e => setEngine(Number(e.target.value))}
              className="w-full bg-white/5 border border-border text-text-pri text-sm font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-primary/40"
            >
              {engines.map(u => <option key={u} value={u} className="bg-bg-sec">Engine {u}</option>)}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Model</label>
            <select
              value={modelKey}
              onChange={e => setModelKey(e.target.value)}
              className="w-full bg-white/5 border border-border text-text-pri text-sm font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-primary/40"
            >
              <option value="rul_rf"  className="bg-bg-sec">Random Forest (Preferred)</option>
              <option value="rul_gb"  className="bg-bg-sec">Gradient Boosting</option>
            </select>
          </div>

          <button
            onClick={run}
            disabled={loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading ? <LoadingSpinner size={16} /> : <Activity size={16} />}
            {loading ? 'Running…' : 'PREDICT RUL'}
          </button>

          {/* Model info */}
          <div className="glass-sm p-3 space-y-1">
            <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest">FD001 Benchmark</p>
            {[
              { k: 'RF MAE',  v: '13.15 cycles' },
              { k: 'RF R²',   v: '0.804'        },
              { k: 'GB MAE',  v: '14.15 cycles' },
            ].map(m => (
              <div key={m.k} className="flex justify-between text-xs font-mono">
                <span className="text-text-sec">{m.k}</span>
                <span className="text-primary">{m.v}</span>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Result */}
        <div className="lg:col-span-3 space-y-4">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div key="result" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                <GlassCard>
                  <div className="flex flex-col md:flex-row items-center gap-8">
                    <RULGauge rul={result.predicted_rul} />
                    <div className="flex-1 space-y-3">
                      <div>
                        <p className="text-xs font-mono text-text-sec">Engine</p>
                        <p className="font-display font-bold text-2xl text-text-pri">ENG-{String(result.engine_unit).padStart(3,'0')}</p>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Predicted RUL</p>
                          <p className="text-xl font-bold text-primary">{formatRUL(result.predicted_rul)}</p>
                        </div>
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Last Cycle</p>
                          <p className="text-xl font-bold text-text-pri">{result.last_cycle}</p>
                        </div>
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Model</p>
                          <p className="text-sm font-mono text-secondary">{result.model}</p>
                        </div>
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Health Status</p>
                          <p className="text-sm font-mono capitalize" style={{ color: healthColor(result.health_status) }}>{result.health_status.replace('_', ' ')}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            ) : (
              <motion.div key="empty" className="glass rounded-2xl p-12 text-center text-text-sec">
                <Activity size={40} className="mx-auto mb-4 opacity-30" />
                <p className="font-display">Select an engine and run prediction</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Recent history chart */}
          {histChartData.length > 0 && (
            <GlassCard>
              <h3 className="font-display text-sm font-semibold text-text-sec uppercase tracking-wide mb-3">Recent Predictions — {dataset}</h3>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={histChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
                  <XAxis dataKey="engine" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#9AAAC2', fontSize: 10 }} />
                  <Tooltip {...TT} />
                  <ReferenceLine y={30} stroke="#FF647C" strokeDasharray="4 4" label={{ value: 'Threshold', fill: '#FF647C', fontSize: 10 }} />
                  <Line type="monotone" dataKey="rul" stroke="#69C9FF" dot={{ r: 3 }} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  )
}
