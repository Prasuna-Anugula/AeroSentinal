import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Cpu, AlertTriangle, Activity, TrendingDown, Zap, BarChart3, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import KpiCard from '@/components/ui/KpiCard'
import GlassCard from '@/components/ui/GlassCard'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { getDashboardSummary, getFleet } from '@/services/api'
import type { DashboardSummary, FleetEngine } from '@/types'
import { healthColor } from '@/utils'

interface Props { dataset: string }

const CHART_TOOLTIP_STYLE = {
  contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 12 },
  labelStyle: { color: '#9AAAC2' },
}

export default function Overview({ dataset }: Props) {
  const [summary,  setSummary]  = useState<DashboardSummary | null>(null)
  const [fleet,    setFleet]    = useState<FleetEngine[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const [slowNote, setSlowNote] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setSummary(null)
    setFleet([])

    // Warn user if it takes more than 4 s (large datasets like FD002/FD004)
    const slowTimer = setTimeout(() => setSlowNote(true), 4000)

    Promise.all([
      getDashboardSummary(dataset),
      getFleet(dataset),
    ]).then(([s, f]) => {
      setSummary(s.data)
      setFleet(f.data)
      setError(null)
    }).catch((err) => {
      setError(err?.response?.data?.detail ?? 'Failed to load dashboard data. Ensure the backend is running.')
    }).finally(() => {
      setLoading(false)
      setSlowNote(false)
      clearTimeout(slowTimer)
    })

    return () => clearTimeout(slowTimer)
  }, [dataset])

  if (loading) return (
    <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
      <LoadingSpinner size={40} />
      <p className="text-text-sec text-sm font-mono">LOADING {dataset} FLEET DATA…</p>
      {slowNote && (
        <p className="text-warning text-xs font-mono text-center max-w-xs">
          {dataset} has more engines than FD001 — first load may take 15–30 s while models score all engines. Results are cached after that.
        </p>
      )}
    </div>
  )

  if (error) return (
    <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
      <AlertTriangle size={40} className="text-danger" />
      <p className="text-danger text-sm font-mono text-center max-w-md">{error}</p>
      <button onClick={() => window.location.reload()} className="btn-ghost text-xs">Retry</button>
    </div>
  )

  const healthDist = summary ? [
    { name: 'Healthy',   value: summary.healthy_engines,   color: '#39D98A' },
    { name: 'Warning',   value: summary.warning_engines,   color: '#FFBF69' },
    { name: 'High Risk', value: summary.high_risk_engines, color: '#FB923C' },
    { name: 'Critical',  value: summary.critical_engines,  color: '#FF647C' },
  ] : []

  const rulBuckets = (() => {
    const buckets: Record<string, number> = { '0-20': 0, '21-50': 0, '51-100': 0, '100+': 0 }
    fleet.forEach(e => {
      const r = e.predicted_rul ?? 0
      if (r <= 20) buckets['0-20']++
      else if (r <= 50) buckets['21-50']++
      else if (r <= 100) buckets['51-100']++
      else buckets['100+']++
    })
    return Object.entries(buckets).map(([name, count]) => ({ name, count }))
  })()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold text-text-pri">Mission Control Overview</h1>
          <p className="text-text-sec text-sm mt-1 font-mono">{summary?.source_label}</p>
        </div>
        <span className="text-xs font-mono px-3 py-1 rounded-full border border-primary/30 text-primary bg-primary/5">
          {dataset} · {summary?.total_engines ?? '—'} engines
        </span>
      </div>

      {/* Non-FD001 caveat */}
      {dataset !== 'FD001' && (
        <div className="flex items-start gap-3 px-4 py-3 rounded-xl bg-warning/8 border border-warning/25">
          <Info size={14} className="text-warning mt-0.5 shrink-0" />
          <p className="text-xs text-warning font-mono">
            Models were trained and evaluated on FD001. Predictions shown for {dataset} use the same trained weights applied to {dataset} sensor data.
            Benchmark metrics (MAE, R², F1, AUC) below are FD001 figures and do not reflect {dataset} accuracy.
          </p>
        </div>
      )}

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        <KpiCard title="Engines"      value={summary?.total_engines ?? '—'}         icon={Cpu}          color="#69C9FF" />
        <KpiCard title="Avg RUL"      value={summary ? `${summary.avg_predicted_rul}` : '—'} sub="cycles" icon={Activity} color="#36E0C1" />
        <KpiCard title="RUL < 30"     value={summary?.engines_rul_below_30 ?? '—'}  icon={TrendingDown} color="#FFBF69" />
        <KpiCard title="Critical"     value={summary?.critical_engines ?? '—'}      icon={AlertTriangle}color="#FF647C" />
        <KpiCard title="High Risk"    value={summary?.engines_high_failure_risk ?? '—'} sub=">70% prob" icon={Zap} color="#FB923C" />
        <KpiCard title="RUL R²"       value={summary ? summary.rul_r2.toFixed(3) : '—'} sub="Random Forest" icon={BarChart3} color="#8B83FF" />
      </div>

      {/* Metrics banner */}
      <GlassCard className="!rounded-xl">
        <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest mb-4">FD001 Existing Training Run — Verified Metrics</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'RUL MAE',    value: summary ? `${summary.rul_mae.toFixed(2)} cycles` : '—', color: '#69C9FF' },
            { label: 'RUL RMSE',   value: summary ? summary.rul_rmse.toFixed(2) : '—',           color: '#69C9FF' },
            { label: 'Failure F1', value: summary ? summary.failure_f1.toFixed(3) : '—',          color: '#36E0C1' },
            { label: 'ROC-AUC',    value: summary ? summary.failure_roc_auc.toFixed(3) : '—',     color: '#36E0C1' },
          ].map(m => (
            <div key={m.label} className="glass-sm p-3">
              <p className="text-xs text-text-sec font-mono">{m.label}</p>
              <p className="text-xl font-bold font-display mt-1" style={{ color: m.color }}>{m.value}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Charts row */}
      <div className="grid md:grid-cols-2 gap-5">
        {/* RUL distribution bar */}
        <GlassCard>
          <h3 className="font-display font-semibold mb-4 text-text-sec text-sm uppercase tracking-wide">RUL Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={rulBuckets}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
              <XAxis dataKey="name" tick={{ fill: '#9AAAC2', fontSize: 11 }} />
              <YAxis tick={{ fill: '#9AAAC2', fontSize: 11 }} />
              <Tooltip {...CHART_TOOLTIP_STYLE} />
              <Bar dataKey="count" fill="#69C9FF" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Health donut */}
        <GlassCard>
          <h3 className="font-display font-semibold mb-4 text-text-sec text-sm uppercase tracking-wide">Fleet Health</h3>
          <div className="flex items-center justify-between">
            <ResponsiveContainer width="55%" height={200}>
              <PieChart>
                <Pie data={healthDist} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" paddingAngle={3}>
                  {healthDist.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip {...CHART_TOOLTIP_STYLE} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2 pr-2">
              {healthDist.map(h => (
                <div key={h.name} className="flex items-center gap-2 text-sm">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: h.color }} />
                  <span className="text-text-sec font-mono text-xs">{h.name}</span>
                  <span className="font-bold ml-auto" style={{ color: h.color }}>{h.value}</span>
                </div>
              ))}
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Production monitoring note */}
      <GlassCard className="border-primary/20">
        <p className="text-xs font-mono text-primary/60 uppercase tracking-widest mb-3">Production Monitoring Summary — FD001 Historical Results</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'RUL Below 30 Cycles', value: summary?.engines_rul_below_30 ?? 24, color: '#FFBF69' },
            { label: 'RUL Below 15 Cycles', value: summary?.engines_rul_below_15 ?? 10, color: '#FF647C' },
            { label: 'Failure Prob > 0.7',  value: summary?.engines_high_failure_risk ?? 26, color: '#FB923C' },
          ].map(m => (
            <div key={m.label} className="text-center">
              <div className="text-4xl font-bold font-display" style={{ color: m.color }}>{m.value}</div>
              <div className="text-xs text-text-sec mt-1 font-mono">{m.label}</div>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  )
}
