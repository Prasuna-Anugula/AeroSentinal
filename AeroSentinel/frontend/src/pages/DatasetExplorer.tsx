import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { Database } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { getDatasets, getLifecycleStats, getRULDistribution } from '@/services/api'
import type { DatasetSummary } from '@/types'

const TT = { contentStyle: { background: '#0B1220', border: '1px solid rgba(148,180,220,0.16)', borderRadius: 10, fontSize: 11 }, labelStyle: { color: '#9AAAC2' } }

interface Props { dataset: string }

export default function DatasetExplorer({ dataset }: Props) {
  const [datasets,   setDatasets]   = useState<DatasetSummary[]>([])
  const [lifecycle,  setLifecycle]  = useState<any | null>(null)
  const [rulDist,    setRulDist]    = useState<any | null>(null)
  const [loading,    setLoading]    = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      getDatasets(),
      getLifecycleStats(dataset),
      getRULDistribution(dataset),
    ]).then(([d, l, r]) => {
      setDatasets(d.data)
      setLifecycle(l.data)
      setRulDist(r.data)
    }).finally(() => setLoading(false))
  }, [dataset])

  if (loading) return <PageLoader />

  // Bucket lifecycle distribution
  const lcBuckets = (() => {
    if (!lifecycle?.distribution) return []
    const vals: number[] = lifecycle.distribution
    const buckets = [0,50,100,150,200,250,300,350]
    const counts  = new Array(buckets.length - 1).fill(0)
    vals.forEach(v => {
      for (let i = 0; i < buckets.length - 1; i++) {
        if (v >= buckets[i] && v < buckets[i+1]) { counts[i]++; break }
      }
    })
    return counts.map((c, i) => ({ range: `${buckets[i]}-${buckets[i+1]}`, count: c }))
  })()

  const rulBuckets = (() => {
    if (!rulDist?.values) return []
    const vals: number[] = rulDist.values
    const buckets = [0,25,50,75,100,150,200,300]
    const counts  = new Array(buckets.length - 1).fill(0)
    vals.forEach(v => {
      for (let i = 0; i < buckets.length - 1; i++) {
        if (v >= buckets[i] && v < buckets[i+1]) { counts[i]++; break }
      }
    })
    return counts.map((c, i) => ({ range: `${buckets[i]}-${buckets[i+1]}`, count: c }))
  })()

  const COMPLEXITY_COLOR: Record<string, string> = { Simple: '#39D98A', Medium: '#FFBF69', Complex: '#FF647C' }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold">Dataset Explorer</h1>
        <p className="text-text-sec text-sm mt-1">NASA C-MAPSS Turbofan Engine Degradation Dataset — all four subsets</p>
      </div>

      {/* Dataset cards */}
      <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
        {datasets.map((d, i) => (
          <motion.div
            key={d.subset}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07 }}
            className={`glass-sm p-4 border ${d.subset === dataset ? 'border-primary/50 bg-primary/5' : 'border-border'}`}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="font-mono font-bold text-primary text-lg">{d.subset}</span>
              <span
                className="text-[10px] font-mono px-2 py-0.5 rounded-full border"
                style={{ color: COMPLEXITY_COLOR[d.complexity], borderColor: COMPLEXITY_COLOR[d.complexity] + '40', background: COMPLEXITY_COLOR[d.complexity] + '15' }}
              >
                {d.complexity}
              </span>
            </div>
            <div className="space-y-1">
              {[
                { label: 'Conditions', value: d.conditions },
                { label: 'Fault modes', value: d.fault_modes },
                { label: 'Train engines', value: d.train_engines },
                { label: 'Test engines', value: d.test_engines },
                { label: 'Train obs', value: d.train_observations.toLocaleString() },
                { label: 'Sensors', value: d.sensor_count },
              ].map(r => (
                <div key={r.label} className="flex justify-between text-xs font-mono">
                  <span className="text-text-sec">{r.label}</span>
                  <span className="text-text-pri">{r.value}</span>
                </div>
              ))}
            </div>
            <div className={`mt-3 text-[10px] font-mono py-0.5 text-center rounded ${d.models_available ? 'text-success bg-success/10' : 'text-text-sec bg-white/5'}`}>
              {d.models_available ? 'MODELS AVAILABLE' : 'DATA ONLY'}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts for selected dataset */}
      <div className="grid md:grid-cols-2 gap-5">
        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-1">Engine Lifecycle Distribution — {dataset}</h3>
          {lifecycle && (
            <p className="text-xs text-text-sec font-mono mb-4">
              Mean: {lifecycle.mean} · Min: {lifecycle.min} · Max: {lifecycle.max} · Std: {lifecycle.std}
            </p>
          )}
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={lcBuckets}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
              <XAxis dataKey="range" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9AAAC2', fontSize: 10 }} />
              <Tooltip {...TT} />
              <Bar dataKey="count" fill="#69C9FF" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>

        <GlassCard>
          <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-1">True RUL Distribution — {dataset}</h3>
          {rulDist && (
            <p className="text-xs text-text-sec font-mono mb-4">
              Mean: {rulDist.mean} · Min: {rulDist.min} · Max: {rulDist.max}
            </p>
          )}
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={rulBuckets}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,180,220,0.08)" />
              <XAxis dataKey="range" tick={{ fill: '#9AAAC2', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9AAAC2', fontSize: 10 }} />
              <Tooltip {...TT} />
              <Bar dataKey="count" fill="#36E0C1" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </GlassCard>
      </div>

      {/* Reference EDA image */}
      <GlassCard>
        <h3 className="font-display font-semibold text-sm text-text-sec uppercase tracking-wide mb-4">Reference — Exploratory Data Analysis (FD001)</h3>
        <p className="text-xs text-text-sec/60 font-mono mb-4">Original visualisation from the training pipeline — lifecycle distribution, sensor degradation, correlation heatmap, operating conditions</p>
        <img src="/static/eda_visualization.png" alt="EDA visualisation" className="rounded-xl w-full object-contain max-h-[500px]" />
      </GlassCard>
    </div>
  )
}
