import { motion } from 'framer-motion'
import { cn } from '@/utils'

interface Props {
  children: React.ReactNode
  className?: string
  hover?: boolean
  animate?: boolean
}

export default function GlassCard({ children, className, hover = true, animate = true }: Props) {
  const Comp = animate ? motion.div : 'div'
  const props = animate ? {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.35, ease: 'easeOut' },
    whileHover: hover ? { y: -2, transition: { duration: 0.15 } } : undefined,
  } : {}

  return (
    <Comp
      {...props}
      className={cn(
        'glass p-5',
        hover && 'cursor-default',
        className
      )}
    >
      {children}
    </Comp>
  )
}
