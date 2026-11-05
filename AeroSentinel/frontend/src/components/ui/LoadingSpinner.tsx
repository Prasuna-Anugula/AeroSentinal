import { motion } from 'framer-motion'
import { cn } from '@/utils'

export default function LoadingSpinner({ className, size = 24 }: { className?: string; size?: number }) {
  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      style={{ width: size, height: size }}
      className={cn('rounded-full border-2 border-primary/20 border-t-primary', className)}
    />
  )
}

export function PageLoader() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[300px] gap-4">
      <LoadingSpinner size={40} />
      <p className="text-text-sec text-sm font-mono">PROCESSING …</p>
    </div>
  )
}
