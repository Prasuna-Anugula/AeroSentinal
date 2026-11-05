import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type { HealthStatus, AnomalySeverity, MaintenancePriority, RiskLevel } from '@/types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function healthLabel(s: HealthStatus): string {
  return { healthy: 'Healthy', warning: 'Warning', high_risk: 'High Risk', critical: 'Critical' }[s] ?? s
}

export function healthColor(s: HealthStatus): string {
  return { healthy: '#39D98A', warning: '#FFBF69', high_risk: '#FB923C', critical: '#FF647C' }[s] ?? '#9AAAC2'
}

export function severityColor(s: AnomalySeverity): string {
  return { normal: '#39D98A', unusual: '#FFBF69', high_anomaly: '#FF647C' }[s] ?? '#9AAAC2'
}

export function priorityColor(p: MaintenancePriority): string {
  return { normal: '#39D98A', medium: '#FFBF69', high: '#FB923C', critical: '#FF647C' }[p] ?? '#9AAAC2'
}

export function riskColor(r: RiskLevel): string {
  return { low: '#39D98A', medium: '#FFBF69', high: '#FF647C' }[r] ?? '#9AAAC2'
}

export function formatRUL(rul: number | null): string {
  if (rul === null || rul === undefined) return '—'
  return `${Math.round(rul)} cycles`
}

export function formatProb(p: number | null): string {
  if (p === null || p === undefined) return '—'
  return `${(p * 100).toFixed(1)}%`
}

export function formatScore(s: number | null): string {
  if (s === null || s === undefined) return '—'
  return s.toFixed(4)
}

export function timeAgo(iso: string): string {
  const d   = new Date(iso)
  const now = Date.now()
  const diff = now - d.getTime()
  if (diff < 60000)  return 'just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return d.toLocaleDateString()
}
