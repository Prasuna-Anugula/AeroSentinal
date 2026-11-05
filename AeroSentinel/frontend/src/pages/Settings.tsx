import GlassCard from '@/components/ui/GlassCard'
import { getCSVReport, getPDFReport } from '@/services/api'

export default function Settings() {
  return (
    <div className="space-y-6 max-w-2xl">
      <h1 className="text-2xl font-display font-bold">System Settings</h1>

      <GlassCard className="space-y-4">
        <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Backend Connection</h3>
        <div className="glass-sm p-3 text-xs font-mono space-y-1">
          <div className="flex justify-between"><span className="text-text-sec">API Base</span><span className="text-primary">http://localhost:8000/api</span></div>
          <div className="flex justify-between"><span className="text-text-sec">Docs</span><a href="/api/docs" target="_blank" className="text-primary hover:underline">/api/docs</a></div>
        </div>
      </GlassCard>

      <GlassCard className="space-y-4">
        <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Report Export — Test</h3>
        <p className="text-xs text-text-sec">Download a sample engine report (Engine 1, FD001)</p>
        <div className="flex gap-3">
          <a href={getCSVReport('FD001', 1)} download className="btn-ghost text-xs">Download CSV</a>
          <a href={getPDFReport('FD001', 1)} download className="btn-primary text-xs">Download PDF</a>
        </div>
      </GlassCard>

      <GlassCard className="space-y-3">
        <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide">Health Thresholds</h3>
        <p className="text-xs text-text-sec/60">AeroSentinel project-defined thresholds. Not certified aviation limits.</p>
        <div className="space-y-2">
          {[
            { label: 'Healthy',   range: '> 100 cycles', color: '#39D98A' },
            { label: 'Warning',   range: '50–100 cycles', color: '#FFBF69' },
            { label: 'High Risk', range: '20–49 cycles', color: '#FB923C' },
            { label: 'Critical',  range: '< 20 cycles',  color: '#FF647C' },
          ].map(t => (
            <div key={t.label} className="flex items-center gap-3 text-xs font-mono">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: t.color }} />
              <span className="text-text-sec w-20">{t.label}</span>
              <span style={{ color: t.color }}>{t.range}</span>
            </div>
          ))}
        </div>
      </GlassCard>

      <GlassCard>
        <h3 className="font-display font-semibold text-text-sec text-xs uppercase tracking-wide mb-3">About AeroSentinel</h3>
        <div className="text-xs font-mono space-y-1.5 text-text-sec">
          <p>Version: 1.0.0</p>
          <p>Dataset: NASA C-MAPSS Turbofan Engine Degradation</p>
          <p>FD001 RUL Model: Random Forest (MAE 13.15 cycles, R² 0.804)</p>
          <p>FD001 Failure Classifier: Gradient Boosting (F1 0.943, AUC 0.998)</p>
          <p className="text-text-sec/40 mt-3">Research-grade ML platform. Not for operational aviation use.</p>
        </div>
      </GlassCard>
    </div>
  )
}
