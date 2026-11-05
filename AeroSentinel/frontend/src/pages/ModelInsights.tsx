import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { BarChart3, CheckCircle, XCircle } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { getModels, getModelComparison } from '@/services/api'
import type { ModelInfo } from '@/types'

const TT = { contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 11 } }

export default function ModelInsights() {
  const [models,     setModels]     = useState<ModelInfo[]>([])
  const [comparison, setComparison] = useState<any | null>(null)
  const [loading,    setLoading]    = useState(true)

  useEffect(() => {
    Promise.all([getModels(), getModelComparison()]).then(([m, c]) => {
      setModels(m.data)
      setComparison(c.data)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return <PageLoader />

  const rulData = comparison?.rul?.map((m: any) => ({
    model: m.model === 'Random Forest' ? 'RF' : 'GB',
    MAE: m.mae,
    RMSE: m.rmse,
    R2: m.r2,
    preferred: m.preferred,
  })) ?? []

  const clsData = comparison?.classification?.map((m: any) => ({
    model: m.model === 'Random Forest' ? 'RF' : 'GB',
    Precision: m.precision,
    Recall: m.recall,
    F1: m.f1,
    AUC: m.roc_auc,
    preferred: m.preferred,
  })) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold">Model Insights</h1>
        <p className="text-text-sec text-sm mt-1 font-mono">FD001 — Existing Training Run · Verified Evaluation Results</p>
      </div>

      {/* Model cards */}
      <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
        {models.map((m, i) => (
          <GlassCard key={m.model_key} className={m.metrics?.f1 === 0.943 || m.metrics?.r2 === 0.804 ? 'border-primary/30' : ''}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono text-text-sec uppercase">{m.model_type}</span>
              {m.is_available
                ? <CheckCircle size={14} className="text-success" />
                : <XCircle size={14} className="text-danger" />}
            </div>
            <p className="font-display font-semibold text-text-pri text-sm mb-3">{m.display_name}</p>
            <div className="space-y-1">
              {Object.entries(m.metrics).map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs font-mono">
                  <span className="text-text-sec">{k.toUpperCase()}</span>
                  <span className="text-primary">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        ))}
      </div>

      {/* RUL comparison charts */}
      <div className="grid md:grid-cols-2 gap-5">
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-1">RUL Regression — MAE & RMSE</h3>
          <p className="text-xs text-text-sec/60 font-mono mb-4">Lower is better. Random Forest preferred for FD001.</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={rulData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
              <XAxis dataKey="model" tick={{ fill: '#9AAAC2', fontSize: 11 }} />
              <YAxis tick={{ fill: '#9AAAC2', fontSize: 11 }} />
              <Tooltip {...TT} />
              <Bar dataKey="MAE"  fill="#69C9FF" radius={[4,4,0,0]} />
              <Bar dataKey="RMSE" fill="#8B83FF" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-1">
            {[
              { label: 'RF MAE',  value: '13.15 cycles', preferred: true },
              { label: 'RF R²',   value: '0.804',        preferred: true },
              { label: 'GB MAE',  value: '14.15 cycles', preferred: false },
              { label: 'GB R²',   value: '0.782',        preferred: false },
            ].map(r => (
              <div key={r.label} className="flex justify-between text-xs font-mono border-b border-border/30 pb-1">
                <span className="text-text-sec">{r.label}</span>
                <span className={r.preferred ? 'text-primary font-bold' : 'text-text-sec'}>{r.value}</span>
              </div>
            ))}
          </div>
        </GlassCard>

        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-1">Failure Classification — F1 & AUC</h3>
          <p className="text-xs text-text-sec/60 font-mono mb-4">Higher is better. Gradient Boosting preferred for FD001.</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={clsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
              <XAxis dataKey="model" tick={{ fill: '#9AAAC2', fontSize: 11 }} />
              <YAxis domain={[0.8, 1.0]} tick={{ fill: '#9AAAC2', fontSize: 11 }} />
              <Tooltip {...TT} />
              <Bar dataKey="F1"  fill="#36E0C1" radius={[4,4,0,0]} />
              <Bar dataKey="AUC" fill="#39D98A" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-1">
            {[
              { label: 'GB F1',        value: '0.943', preferred: true },
              { label: 'GB ROC-AUC',   value: '0.998', preferred: true },
              { label: 'GB Recall',    value: '1.000', preferred: true },
              { label: 'RF F1',        value: '0.941', preferred: false },
            ].map(r => (
              <div key={r.label} className="flex justify-between text-xs font-mono border-b border-border/30 pb-1">
                <span className="text-text-sec">{r.label}</span>
                <span className={r.preferred ? 'text-accent font-bold' : 'text-text-sec'}>{r.value}</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Production monitoring */}
      <GlassCard className="border-primary/20">
        <p className="text-xs font-mono text-primary/60 uppercase tracking-widest mb-4">FD001 — Historical Production Monitoring Results (100 test engines)</p>
        <div className="grid grid-cols-3 gap-6 mb-6">
          {[
            { label: 'RUL Below 30 Cycles', value: 24, color: '#FFBF69' },
            { label: 'RUL Below 15 Cycles', value: 10, color: '#FF647C' },
            { label: 'Failure Prob > 0.7',  value: 26, color: '#FB923C' },
          ].map(m => (
            <div key={m.label} className="text-center">
              <div className="text-5xl font-bold font-display" style={{ color: m.color }}>{m.value}</div>
              <div className="text-xs text-text-sec mt-1 font-mono">{m.label}</div>
            </div>
          ))}
        </div>
        <img src="/static/production_monitoring_dashboard.png" alt="Production monitoring dashboard" className="rounded-xl w-full object-contain max-h-[400px]" />
      </GlassCard>

      {/* Reference images */}
      <div className="grid md:grid-cols-2 gap-5">
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Reference — RUL Model Results (FD001)</h3>
          <img src="/static/rul_model_results.png" alt="RUL model results" className="rounded-xl w-full" />
        </GlassCard>
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Reference — Failure Model Results (FD001)</h3>
          <img src="/static/failure_model_results.png" alt="Failure model results" className="rounded-xl w-full" />
        </GlassCard>
      </div>
    </div>
  )
}
