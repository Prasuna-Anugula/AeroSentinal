import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Radio } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { useToast } from '@/components/ui/Toast'
import { getFeatureImportance, getEngines, getEngineExplanation } from '@/services/api'

const TT = { contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 11 } }

interface Props { dataset: string }

export default function SensorIntelligence({ dataset }: Props) {
  const { toast } = useToast()
  const [modelKey,    setModelKey]    = useState('rul_rf')
  const [importance,  setImportance]  = useState<any[]>([])
  const [engines,     setEngines]     = useState<number[]>([])
  const [engine,      setEngine]      = useState(1)
  const [explanation, setExplanation] = useState<any | null>(null)
  const [loadingGlobal, setLoadingGlobal] = useState(true)
  const [loadingLocal,  setLoadingLocal]  = useState(false)

  useEffect(() => {
    setLoadingGlobal(true)
    getFeatureImportance(modelKey, 20).then(r => {
      setImportance(r.data.items ?? [])
    }).finally(() => setLoadingGlobal(false))
    getEngines(dataset).then(r => {
      const u: number[] = r.data.engines ?? []
      setEngines(u)
      if (u.length) setEngine(u[0])
    })
  }, [modelKey, dataset])

  async function runExplanation() {
    setLoadingLocal(true)
    try {
      const r = await getEngineExplanation(dataset, engine, modelKey)
      setExplanation(r.data)
      toast('Local explanation generated', 'success')
    } catch (e: any) {
      toast('Explanation failed', 'error')
    } finally {
      setLoadingLocal(false)
    }
  }

  // Get top sensor-level importances (aggregate by base sensor name)
  const sensorAgg = importance.reduce((acc, item) => {
    const sensor = item.feature.match(/sensor\d+/)?.[0]
    if (sensor) {
      acc[sensor] = (acc[sensor] ?? 0) + item.importance
    }
    return acc
  }, {} as Record<string, number>)
  const topSensors = Object.entries(sensorAgg)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([sensor, imp]) => ({ sensor, importance: imp }))

  const topFeatures = importance.slice(0, 20)

  const COLORS = ['#69C9FF','#8B83FF','#36E0C1','#FFBF69','#39D98A','#FF647C','#FB923C','#A78BFA','#34D399','#F472B6']

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold">Sensor Intelligence</h1>
        <p className="text-text-sec text-sm mt-1">Feature importance and per-engine explanation — which sensors drive predictions?</p>
      </div>

      {/* Model selector */}
      <div className="flex gap-3">
        {[
          { key: 'rul_rf',  label: 'RUL — Random Forest' },
          { key: 'rul_gb',  label: 'RUL — Gradient Boosting' },
          { key: 'fail_gb', label: 'Failure — Gradient Boosting' },
          { key: 'fail_rf', label: 'Failure — Random Forest' },
        ].map(m => (
          <button
            key={m.key}
            onClick={() => setModelKey(m.key)}
            className={`text-xs font-mono px-3 py-1.5 rounded-lg border transition-colors ${modelKey === m.key ? 'border-primary/50 bg-primary/10 text-primary' : 'border-border text-text-sec hover:border-primary/30'}`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        {/* Global feature importance */}
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Top 20 Features — {modelKey}</h3>
          <p className="text-xs text-text-sec/60 font-mono mb-3">Global feature importance from trained model. Importance reflects contribution to prediction accuracy, not physical causality.</p>
          {loadingGlobal ? <div className="flex justify-center py-8"><LoadingSpinner /></div> : (
            <ResponsiveContainer width="100%" height={380}>
              <BarChart data={topFeatures} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
                <XAxis type="number" tick={{ fill: '#9AAAC2', fontSize: 9 }} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#9AAAC2', fontSize: 9 }} width={130} />
                <Tooltip {...TT} />
                <Bar dataKey="importance" radius={[0,4,4,0]}>
                  {topFeatures.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </GlassCard>

        {/* Sensor aggregated importance */}
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Sensor Contribution Ranking</h3>
          <p className="text-xs text-text-sec/60 font-mono mb-3">Aggregated importance across all engineered features for each base sensor channel.</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topSensors}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
              <XAxis dataKey="sensor" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9AAAC2', fontSize: 10 }} />
              <Tooltip {...TT} />
              <Bar dataKey="importance" fill="#8B83FF" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>

          {/* Local explanation */}
          <div className="mt-5 pt-4 border-t border-border">
            <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-3">Local Engine Explanation</h3>
            <div className="flex gap-2 mb-3">
              <select value={engine} onChange={e => setEngine(Number(e.target.value))}
                className="flex-1 bg-white/5 border border-border text-text-pri text-xs font-mono rounded-lg px-2 py-1.5 focus:outline-none focus:border-primary/40">
                {engines.slice(0, 30).map(u => <option key={u} value={u} className="bg-bg-sec">Engine {u}</option>)}
              </select>
              <button onClick={runExplanation} disabled={loadingLocal}
                className="btn-ghost flex items-center gap-1.5 text-xs px-3">
                {loadingLocal ? <LoadingSpinner size={12} /> : <Radio size={12} />}
                Explain
              </button>
            </div>
            {explanation && (
              <div className="space-y-1.5">
                <p className="text-xs font-mono text-text-sec/60">Prediction: {explanation.prediction?.toFixed(2)}</p>
                {explanation.top_features?.slice(0,8).map((f: any, i: number) => (
                  <div key={f.feature} className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-text-sec w-4">{i+1}</span>
                    <div className="flex-1 bg-white/5 rounded h-1.5">
                      <div
                        className="h-full rounded transition-all"
                        style={{ width: `${(f.contribution / (explanation.top_features[0]?.contribution ?? 1)) * 100}%`, background: COLORS[i % COLORS.length] }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-text-sec w-28 truncate">{f.feature}</span>
                    <span className="text-[10px] font-mono" style={{ color: COLORS[i % COLORS.length] }}>{f.contribution.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </GlassCard>
      </div>

      {/* Reference PNGs */}
      <div className="grid md:grid-cols-2 gap-5">
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Reference — RUL Feature Importance (FD001)</h3>
          <img src="/static/feature_importance_rul.png" alt="Feature importance RUL" className="rounded-xl w-full object-contain" />
        </GlassCard>
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Reference — Failure Classifier Feature Importance (FD001)</h3>
          <img src="/static/feature_importance_failure.png" alt="Feature importance failure" className="rounded-xl w-full object-contain" />
        </GlassCard>
      </div>
    </div>
  )
}
