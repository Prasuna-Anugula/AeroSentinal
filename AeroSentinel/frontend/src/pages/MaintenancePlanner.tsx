import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Wrench, CheckCircle, Eye, RefreshCw } from 'lucide-react'
import GlassCard from '@/components/ui/GlassCard'
import StatusBadge from '@/components/ui/StatusBadge'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { useToast } from '@/components/ui/Toast'
import { getEngines, generateRecommendation, getMaintenanceHistory, updateMaintenanceStatus } from '@/services/api'
import type { MaintenanceRecommendation, MaintenancePriority } from '@/types'
import { priorityColor } from '@/utils'

interface Props { dataset: string }

const PRIORITY_ICON: Record<string, string> = {
  critical: '🔴',
  high:     '🟠',
  medium:   '🟡',
  normal:   '🟢',
}

export default function MaintenancePlanner({ dataset }: Props) {
  const { toast } = useToast()
  const [engines,  setEngines]  = useState<number[]>([])
  const [engine,   setEngine]   = useState(1)
  const [history,  setHistory]  = useState<MaintenanceRecommendation[]>([])
  const [loading,  setLoading]  = useState(false)
  const [filter,   setFilter]   = useState<string>('all')

  const loadHistory = () => {
    getMaintenanceHistory(100).then(r => setHistory(r.data))
  }

  useEffect(() => {
    getEngines(dataset).then(r => {
      const u: number[] = r.data.engines ?? []
      setEngines(u)
      if (u.length) setEngine(u[0])
    })
    loadHistory()
  }, [dataset])

  async function generate() {
    setLoading(true)
    try {
      await generateRecommendation(dataset, engine)
      loadHistory()
      toast('Maintenance recommendation generated', 'success')
    } catch (e: any) {
      toast(e.response?.data?.detail ?? 'Failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  async function updateStatus(id: number, status: string) {
    try {
      await updateMaintenanceStatus(id, status)
      loadHistory()
      toast(`Status updated to ${status}`, 'success')
    } catch {
      toast('Update failed', 'error')
    }
  }

  const filtered = filter === 'all' ? history : history.filter(h => h.priority === filter || h.status === filter)

  const counts = history.reduce((acc, h) => {
    acc[h.priority] = (acc[h.priority] ?? 0) + 1
    return acc
  }, {} as Record<string, number>)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold">Maintenance Planner</h1>
        <p className="text-text-sec text-sm mt-1">Deterministic rule-based recommendations derived from RUL, failure probability, and anomaly scores</p>
      </div>

      <div className="grid lg:grid-cols-4 gap-5">
        {/* Generator */}
        <GlassCard className="lg:col-span-1 space-y-4">
          <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Generate Recommendation</h3>

          <div className="space-y-1">
            <label className="text-xs font-mono text-text-sec">Engine Unit</label>
            <select value={engine} onChange={e => setEngine(Number(e.target.value))}
              className="w-full bg-white/5 border border-border text-text-pri text-xs font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-primary/40">
              {engines.map(u => <option key={u} value={u} className="bg-bg-sec">Engine {u}</option>)}
            </select>
          </div>

          <button onClick={generate} disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2 text-sm">
            {loading ? <LoadingSpinner size={16} /> : <Wrench size={16} />}
            {loading ? 'Generating…' : 'GENERATE'}
          </button>

          <div className="glass-sm p-3 space-y-2">
            <p className="text-xs font-mono text-text-sec/60 uppercase tracking-widest">Rule Summary</p>
            <div className="space-y-1.5 text-xs font-mono text-text-sec">
              <div>🔴 RUL &lt; 15 → Critical</div>
              <div>🟠 RUL &lt; 30 → High</div>
              <div>🟠 Prob &gt; 0.7 → High</div>
              <div>🟠 High Anomaly → High</div>
              <div>🟡 Unusual → Medium</div>
              <div>🟢 No flags → Normal</div>
            </div>
            <p className="text-[10px] text-text-sec/40 mt-1">AeroSentinel decision-support rules. Not certified procedures.</p>
          </div>

          {/* Priority counts */}
          <div className="space-y-1">
            {['critical','high','medium','normal'].map(p => (
              <div key={p} className="flex justify-between text-xs font-mono">
                <span style={{ color: priorityColor(p as MaintenancePriority) }}>{PRIORITY_ICON[p]} {p}</span>
                <span className="text-text-sec">{counts[p] ?? 0}</span>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* History */}
        <div className="lg:col-span-3 space-y-4">
          {/* Filters */}
          <div className="flex gap-2 flex-wrap">
            {['all','critical','high','medium','normal','pending','reviewed','resolved'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`text-xs font-mono px-3 py-1 rounded-full border transition-colors ${filter === f ? 'border-primary/50 bg-primary/10 text-primary' : 'border-border text-text-sec hover:border-primary/30'}`}
              >
                {f}
              </button>
            ))}
            <button onClick={loadHistory} className="ml-auto text-xs font-mono text-text-sec hover:text-primary transition-colors flex items-center gap-1">
              <RefreshCw size={11} /> refresh
            </button>
          </div>

          <div className="space-y-3">
            <AnimatePresence>
              {filtered.length === 0 && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass rounded-2xl p-10 text-center text-text-sec">
                  <Wrench size={36} className="mx-auto mb-3 opacity-30" />
                  <p className="font-display">No recommendations yet. Generate one above.</p>
                </motion.div>
              )}
              {filtered.map(rec => (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-sm p-4"
                >
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{PRIORITY_ICON[rec.priority]}</span>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-display font-semibold text-text-pri">
                            ENG-{String(rec.engine_unit).padStart(3,'0')}
                          </span>
                          <span className="text-xs font-mono text-text-sec">{rec.dataset_subset}</span>
                          <StatusBadge value={rec.priority} />
                          <StatusBadge value={rec.status} />
                        </div>
                        <p className="text-sm text-text-sec mt-1">{rec.recommendation}</p>
                      </div>
                    </div>
                    <div className="flex gap-2 shrink-0">
                      {rec.status === 'pending' && (
                        <button onClick={() => updateStatus(rec.id, 'reviewed')}
                          className="text-xs font-mono px-2 py-1 rounded border border-warning/30 text-warning hover:bg-warning/10 flex items-center gap-1">
                          <Eye size={11} /> Review
                        </button>
                      )}
                      {rec.status === 'reviewed' && (
                        <button onClick={() => updateStatus(rec.id, 'resolved')}
                          className="text-xs font-mono px-2 py-1 rounded border border-success/30 text-success hover:bg-success/10 flex items-center gap-1">
                          <CheckCircle size={11} /> Resolve
                        </button>
                      )}
                    </div>
                  </div>
                  {/* Evidence */}
                  {rec.evidence && (
                    <div className="mt-3 flex flex-wrap gap-3 pl-8">
                      {Object.entries(rec.evidence).filter(([,v]) => v !== null).map(([k,v]) => (
                        <span key={k} className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 border border-border text-text-sec">
                          {k}: <span className="text-primary">{typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                  <p className="text-[10px] text-text-sec/40 font-mono mt-2 pl-8">{new Date(rec.created_at).toLocaleString()}</p>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}
