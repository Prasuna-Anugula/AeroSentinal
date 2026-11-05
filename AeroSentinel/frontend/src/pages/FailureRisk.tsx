import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import StatusBadge from '@/components/ui/StatusBadge'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { useToast } from '@/components/ui/Toast'
import { getEngines, predictFailure, getFleet } from '@/services/api'
import type { FailurePrediction, RiskLevel } from '@/types'
import { riskColor, formatProb } from '@/utils'

interface Props { dataset: string }

function ProbGauge({ prob }: { prob: number }) {
  const risk: RiskLevel = prob >= 0.7 ? 'high' : prob >= 0.4 ? 'medium' : 'low'
  const color = riskColor(risk)
  const pct   = prob

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-40 h-40">
        <svg viewBox="0 0 200 200" className="w-full h-full -rotate-90">
          <circle cx="100" cy="100" r="75" fill="none" stroke="rgba(148,180,220,0.1)" strokeWidth="14" />
          <motion.circle
            cx="100" cy="100" r="75"
            fill="none" stroke={color} strokeWidth="14"
            strokeDasharray={`${pct * 471.2} 471.2`}
            strokeLinecap="round"
            initial={{ strokeDasharray: '0 471.2' }}
            animate={{ strokeDasharray: `${pct * 471.2} 471.2` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            key={prob}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-3xl font-bold font-display"
            style={{ color }}
          >
            {(prob * 100).toFixed(1)}%
          </motion.span>
          <span className="text-xs font-mono text-text-sec">fail prob</span>
        </div>
      </div>
      <StatusBadge value={risk} label={`Risk: ${risk.toUpperCase()}`} />
    </div>
  )
}

export default function FailureRisk({ dataset }: Props) {
  const { toast } = useToast()
  const [engines,  setEngines]  = useState<number[]>([])
  const [engine,   setEngine]   = useState(1)
  const [modelKey, setModelKey] = useState('fail_gb')
  const [result,   setResult]   = useState<FailurePrediction | null>(null)
  const [loading,  setLoading]  = useState(false)
  const [fleet,    setFleet]    = useState<any[]>([])

  useEffect(() => {
    getEngines(dataset).then(r => {
      const u: number[] = r.data.engines ?? []
      setEngines(u)
      if (u.length) setEngine(u[0])
    })
    getFleet(dataset).then(r => setFleet(r.data))
  }, [dataset])

  async function run() {
    setLoading(true)
    try {
      const r = await predictFailure(dataset, engine, modelKey)
      setResult(r.data)
      toast('Failure risk assessment complete', 'success')
    } catch (e: any) {
      toast(e.response?.data?.detail ?? 'Failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const highRisk = fleet.filter(e => (e.failure_probability ?? 0) > 0.7)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold">Failure Risk Prediction</h1>
        <p className="text-text-sec text-sm mt-1">Binary classification — will this engine fail within 30 cycles? (FD001 F1: 0.943, ROC-AUC: 0.998)</p>
      </div>

      <div className="grid lg:grid-cols-5 gap-5">
        {/* Controls */}
        <GlassCard className="lg:col-span-2 space-y-4">
          <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Assessment Parameters</h3>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Engine Unit</label>
            <select value={engine} onChange={e => setEngine(Number(e.target.value))}
              className="w-full bg-white/5 border border-border text-text-pri text-sm font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-primary/40">
              {engines.map(u => <option key={u} value={u} className="bg-bg-sec">Engine {u}</option>)}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Classifier</label>
            <select value={modelKey} onChange={e => setModelKey(e.target.value)}
              className="w-full bg-white/5 border border-border text-text-pri text-sm font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-primary/40">
              <option value="fail_gb" className="bg-bg-sec">Gradient Boosting (Preferred)</option>
              <option value="fail_rf" className="bg-bg-sec">Random Forest</option>
            </select>
          </div>

          <button onClick={run} disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
            {loading ? <LoadingSpinner size={16} /> : <AlertTriangle size={16} />}
            {loading ? 'Assessing…' : 'ASSESS FAILURE RISK'}
          </button>

          <div className="glass-sm p-3 space-y-2">
            <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest">FD001 Classifier Benchmark</p>
            {[
              { k: 'GB F1',        v: '0.943' },
              { k: 'GB ROC-AUC',   v: '0.998' },
              { k: 'GB Recall',    v: '1.000' },
              { k: 'RF F1',        v: '0.941' },
            ].map(m => (
              <div key={m.k} className="flex justify-between text-xs font-mono">
                <span className="text-text-sec">{m.k}</span>
                <span className="text-accent">{m.v}</span>
              </div>
            ))}
          </div>

          <div className="glass-sm p-3">
            <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest mb-2">Risk Thresholds</p>
            {[
              { label: 'High Risk', threshold: '≥ 70%', color: '#FF647C' },
              { label: 'Medium',    threshold: '40–70%', color: '#FFBF69' },
              { label: 'Low',       threshold: '< 40%',  color: '#39D98A' },
            ].map(t => (
              <div key={t.label} className="flex items-center gap-2 text-xs font-mono py-0.5">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ background: t.color }} />
                <span className="text-text-sec">{t.label}</span>
                <span className="ml-auto" style={{ color: t.color }}>{t.threshold}</span>
              </div>
            ))}
            <p className="text-[10px] text-text-sec/40 mt-2">AeroSentinel project-defined thresholds. Not certified aviation limits.</p>
          </div>
        </GlassCard>

        {/* Result */}
        <div className="lg:col-span-3 space-y-4">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div key="result" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
                <GlassCard>
                  <div className="flex flex-col md:flex-row items-center gap-8">
                    <ProbGauge prob={result.failure_probability} />
                    <div className="flex-1 space-y-3">
                      <h2 className="font-display font-bold text-2xl">ENG-{String(result.engine_unit).padStart(3,'0')}</h2>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Failure Probability</p>
                          <p className="text-xl font-bold text-warning">{formatProb(result.failure_probability)}</p>
                        </div>
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Classification</p>
                          <p className="text-lg font-bold" style={{ color: result.failure_predicted ? '#FF647C' : '#39D98A' }}>
                            {result.failure_predicted ? 'FAILURE' : 'NO FAILURE'}
                          </p>
                        </div>
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Risk Level</p>
                          <StatusBadge value={result.risk_level} />
                        </div>
                        <div className="glass-sm p-3">
                          <p className="text-xs text-text-sec font-mono">Classifier</p>
                          <p className="text-sm font-mono text-secondary">{result.model}</p>
                        </div>
                      </div>
                      {result.failure_probability >= 0.7 && (
                        <div className="flex items-start gap-3 p-3 rounded-xl bg-danger/10 border border-danger/30">
                          <AlertTriangle size={16} className="text-danger mt-0.5 shrink-0" />
                          <p className="text-sm text-danger">High failure risk detected. Failure probability exceeds the 0.70 monitoring threshold. Consider scheduling an inspection.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            ) : (
              <motion.div key="empty" className="glass rounded-2xl p-12 text-center text-text-sec">
                <AlertTriangle size={40} className="mx-auto mb-4 opacity-30" />
                <p className="font-display">Select an engine and assess failure risk</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Fleet high-risk summary */}
          {highRisk.length > 0 && (
            <GlassCard>
              <h3 className="font-display text-sm font-semibold text-danger uppercase tracking-wide mb-3">
                High Risk Engines ({highRisk.length}) — Probability &gt; 70%
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {highRisk.slice(0, 9).map(e => (
                  <div key={e.engine_unit} className="glass-sm p-2 flex items-center justify-between">
                    <span className="text-xs font-mono text-text-pri">ENG-{String(e.engine_unit).padStart(3,'0')}</span>
                    <span className="text-xs font-mono text-danger font-bold">{formatProb(e.failure_probability)}</span>
                  </div>
                ))}
              </div>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  )
}
