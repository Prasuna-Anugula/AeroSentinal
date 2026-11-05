import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Cpu, Activity, AlertTriangle, Zap,
  Radio, Wrench, Database, BarChart3, Settings,
  ChevronLeft, ChevronRight, Gauge
} from 'lucide-react'
import { cn } from '@/utils'

const NAV = [
  { path: '/app/overview',    label: 'Overview',         icon: LayoutDashboard },
  { path: '/app/engines',     label: 'Engine Monitor',   icon: Cpu },
  { path: '/app/rul',         label: 'RUL Prediction',   icon: Activity },
  { path: '/app/failure',     label: 'Failure Risk',     icon: AlertTriangle },
  { path: '/app/anomaly',     label: 'Anomaly Detection',icon: Zap },
  { path: '/app/sensors',     label: 'Sensor Intelligence', icon: Radio },
  { path: '/app/maintenance', label: 'Maintenance',      icon: Wrench },
  { path: '/app/datasets',    label: 'Dataset Explorer', icon: Database },
  { path: '/app/models',      label: 'Model Insights',   icon: BarChart3 },
  { path: '/app/settings',    label: 'Settings',         icon: Settings },
]

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <motion.aside
      animate={{ width: collapsed ? 64 : 220 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="relative flex flex-col h-screen bg-bg-sec border-r border-border shrink-0 overflow-hidden"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-border">
        <div className="w-8 h-8 shrink-0 rounded-lg bg-primary/20 flex items-center justify-center">
          <Gauge size={16} className="text-primary" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-display font-bold text-sm text-text-pri whitespace-nowrap"
            >
              AeroSentinel
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 overflow-y-auto overflow-x-hidden">
        {NAV.map(({ path, label, icon: Icon }) => {
          const active = location.pathname.startsWith(path)
          return (
            <NavLink
              key={path}
              to={path}
              title={collapsed ? label : undefined}
              className={cn(
                'flex items-center gap-3 px-4 py-2.5 mx-2 rounded-xl text-sm transition-all duration-150 mb-0.5',
                active
                  ? 'bg-primary/15 text-primary font-medium'
                  : 'text-text-sec hover:bg-white/5 hover:text-text-pri'
              )}
            >
              <Icon size={16} className="shrink-0" />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="whitespace-nowrap"
                  >
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          )
        })}
      </nav>

      {/* Collapse button */}
      <button
        onClick={() => setCollapsed(c => !c)}
        className="absolute -right-3 top-8 w-6 h-6 rounded-full bg-bg-sec border border-border flex items-center justify-center text-text-sec hover:text-primary z-10 transition-colors"
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>

      {/* Bottom tag */}
      <div className="px-4 py-3 border-t border-border">
        <AnimatePresence>
          {!collapsed && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-[10px] font-mono text-text-sec/50"
            >
              MISSION CONTROL v1.0
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </motion.aside>
  )
}
