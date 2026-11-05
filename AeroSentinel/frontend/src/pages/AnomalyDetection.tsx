import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from 'recharts'
import { Zap } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import StatusBadge from '@/components/ui/StatusBadge'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { useToast } from '@/components/ui/Toast'
import { getEngines, detectAnomaly, getAnomalySummary } from '@/services/api'
import type { AnomalyResult, AnomalySeverity } from '@/types'
import { severityColor } from '@/utils'

const TT = { contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 11 } }

interface Props { dataset: string }

export default function AnomalyDetection({ dataset }: Props) {
  const { toast } = useToast()
  const [engines,  setEngines]  = useState<number[]>([])
  const [engine,   setEngine]   = useState(1)
  const [result,   setResult]   = useState<AnomalyResult | null>(null)
  const [summary,  setSummary]  = useState<any[]>([])
  const [loading,  setLoading]  = useState(false)

  useEffect(() => {
    getEngines(dataset).then(r => {
      const u: number[] = r.data.engines ?? []
      setEngines(u)
      if (u.length) setEngine(u[0])
    })
    getAnomalySummary(dataset).then(r => setSummary(r.data)).catch(() => {})
  }, [dataset])

  async function run() {
    setLoading(true)
    try {
      const r = await detectAnomaly(dataset, engine)
      setResult(r.data)
      toast('Anomaly detection complete', 'success')
    } catch (e: any) {
      toast(e.response?.data?.detail ?? 'Detection failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const chartData = result?.cycles.map(c => ({
    cycle: c.cycle,
    score: c.anomaly_score,
    anomaly: c.is_anomaly ? c.anomaly_score : null,
  })) ?? []

  const summaryBars = (() => {
    const sev: Record<string, number> = { normal: 0, unusual: 0, high_anomaly: 0 }
    summary.forEach(e => { if (e.severity in sev) sev[e.severity]++ })
    return Object.entries(sev).map(([s, c]) => ({ severity: s, count: c }))
  })()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold">Anomaly Detection</h1>
        <p className="text-text-sec text-sm mt-1">Isolation Forest detects abnormal sensor patterns — score more negative = more anomalous</p>
      </div>

      <div className="grid lg:grid-cols-5 gap-5">
        {/* Controls */}
        <GlassCard className="lg:col-span-2 space-y-4">
          <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Detection Parameters</h3>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Engine Unit</label>
            <select value={engine} onChange={e => setEngine(Number(e.target.value))}
              className="w-full bg-white/5 border border-border text-text-pri text-sm font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-primary/40">
              {engines.map(u => <option key={u} value={u} className="bg-bg-sec">Engine {u}</option>)}
            </select>
          </div>

          <button onClick={run} disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
            {loading ? <LoadingSpinner size={16} /> : <Zap size={16} />}
            {loading ? 'Detecting…' : 'DETECT ANOMALIES'}
          </button>

          <div className="glass-sm p-3 space-y-1">
            <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest mb-2">Severity Thresholds</p>
            {[
              { label: 'Normal',      range: '> -0.10',  color: '#39D98A' },
              { label: 'Unusual',     range: '-0.10 to -0.20', color: '#FFBF69' },
              { label: 'High Anomaly',range: '< -0.20',  color: '#FF647C' },
            ].map(t => (
              <div key={t.label} className="flex items-center gap-2 text-xs font-mono">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ background: t.color }} />
                <span className="text-text-sec flex-1">{t.label}</span>
                <span style={{ color: t.color }}>{t.range}</span>
              </div>
            ))}
            <p className="text-[10px] text-text-sec/40 mt-2">Anomaly scores are not failure probabilities.</p>
          </div>

          {/* Fleet severity summary */}
          {summaryBars.some(b => b.count > 0) && (
            <div className="glass-sm p-3">
              <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest mb-2">Fleet Summary — {dataset}</p>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={summaryBars} layout="vertical">
                  <XAxis type="number" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
                  <YAxis type="category" dataKey="severity" tick={{ fill: '#9AAAC2', fontSize: 10 }} width={80} />
                  <Tooltip {...TT} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {summaryBars.map((b, i) => (
                      <Cell key={i} fill={severityColor(b.severity as AnomalySeverity)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </GlassCard>

        {/* Result */}
        <div className="lg:col-span-3 space-y-4">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div key="result" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                <GlassCard>
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h2 className="font-display font-bold text-xl">ENG-{String(result.engine_unit).padStart(3,'0')}</h2>
                      <p className="text-xs text-text-sec font-mono">{result.total_cycles} cycles analysed · {result.anomaly_count} anomalous cycles</p>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-bold font-mono" style={{ color: severityColor(result.severity) }}>
                        {result.overall_score.toFixed(4)}
                      </p>
                      <StatusBadge value={result.severity} />
                    </div>
                  </div>

                  <h3 className="font-display text-xs font-semibold text-text-sec uppercase tracking-wide mb-2">Anomaly Score Over Cycles</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#69C9FF" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#69C9FF" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
                      <XAxis dataKey="cycle" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#9AAAC2', fontSize: 10 }} />
                      <Tooltip {...TT} />
                      <ReferenceLine y={-0.10} stroke="#FFBF69" strokeDasharray="4 4" />
                      <ReferenceLine y={-0.20} stroke="#FF647C" strokeDasharray="4 4" />
                      <Area type="monotone" dataKey="score" stroke="#69C9FF" fill="url(#scoreGrad)" strokeWidth={1.5} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </GlassCard>

                {/* Top anomalous cycles */}
                <GlassCard>
                  <h3 className="font-display text-sm font-semibold text-text-sec uppercase tracking-wide mb-3">Most Anomalous Cycles</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs font-mono">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 px-2 text-text-sec">Cycle</th>
                          <th className="text-left py-2 px-2 text-text-sec">Score</th>
                          <th className="text-left py-2 px-2 text-text-sec">Severity</th>
                          <th className="text-left py-2 px-2 text-text-sec">Top Sensors</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...result.cycles]
                          .sort((a, b) => a.anomaly_score - b.anomaly_score)
                          .slice(0, 8)
                          .map(c => (
                          <tr key={c.cycle} className="border-b border-border/30">
                            <td className="py-1.5 px-2 text-primary">{c.cycle}</td>
                            <td className="py-1.5 px-2" style={{ color: severityColor(c.severity) }}>{c.anomaly_score.toFixed(4)}</td>
                            <td className="py-1.5 px-2"><StatusBadge value={c.severity} /></td>
                            <td className="py-1.5 px-2 text-text-sec">{c.top_sensors.join(', ')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>
              </motion.div>
            ) : (
              <motion.div key="empty" className="glass rounded-2xl p-12 text-center text-text-sec">
                <Zap size={40} className="mx-auto mb-4 opacity-30" />
                <p className="font-display">Select an engine and run detection</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
