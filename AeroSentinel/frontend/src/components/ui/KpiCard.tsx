import { motion } from 'framer-motion'
import { cn } from '@/utils'
import type { LucideIcon } from 'lucide-react'

interface Props {
  title: string
  value: string | number
  sub?: string
  icon?: LucideIcon
  color?: string
  accent?: string
  className?: string
}

export default function KpiCard({ title, value, sub, icon: Icon, color = '#69C9FF', className }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn('glass-sm p-4 flex flex-col gap-2', className)}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-text-sec uppercase tracking-widest">{title}</span>
        {Icon && (
          <span className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${color}18` }}>
            <Icon size={16} style={{ color }} />
          </span>
        )}
      </div>
      <div className="text-3xl font-display font-bold" style={{ color }}>{value}</div>
      {sub && <div className="text-xs text-text-sec">{sub}</div>}
    </motion.div>
  )
}
