import { cn } from '@/utils'
import type { HealthStatus, AnomalySeverity, MaintenancePriority, RiskLevel } from '@/types'

type BadgeVariant = HealthStatus | AnomalySeverity | MaintenancePriority | RiskLevel | string

const MAP: Record<string, string> = {
  healthy:      'status-healthy',
  warning:      'status-warning',
  high_risk:    'status-high_risk',
  critical:     'status-critical',
  normal:       'status-normal',
  unusual:      'status-unusual',
  high_anomaly: 'status-high_anomaly',
  low:          'status-healthy',
  medium:       'status-warning',
  high:         'status-high_risk',
  pending:      'text-text-sec bg-white/5 border border-white/10',
  reviewed:     'status-warning',
  resolved:     'status-healthy',
}

const LABELS: Record<string, string> = {
  healthy:      'Healthy',
  warning:      'Warning',
  high_risk:    'High Risk',
  critical:     'Critical',
  normal:       'Normal',
  unusual:      'Unusual',
  high_anomaly: 'High Anomaly',
  low:          'Low',
  medium:       'Medium',
  high:         'High',
  pending:      'Pending',
  reviewed:     'Reviewed',
  resolved:     'Resolved',
}

interface Props {
  value: BadgeVariant
  className?: string
  label?: string
}

export default function StatusBadge({ value, className, label }: Props) {
  const cls = MAP[value] ?? 'text-text-sec bg-white/5 border border-white/10'
  return (
    <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium font-mono', cls, className)}>
      {label ?? LABELS[value] ?? value}
    </span>
  )
}
