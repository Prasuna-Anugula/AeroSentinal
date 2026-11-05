import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Cpu, Activity, Zap, Radio, Wrench, BarChart3, ChevronDown } from 'lucide-react'

const TICKER = ['SYSTEM: LOCAL', 'DATASET: NASA C-MAPSS', 'SENSOR CHANNELS: 21', 'DATASET SUBSETS: 04', 'ML PIPELINE: READY']

const FEATURES = [
  { icon: Activity,  title: 'RUL Estimation',         desc: 'Predict remaining operating cycles using Random Forest regression trained on run-to-failure trajectories.' },
  { icon: AlertIcon, title: 'Failure Risk Prediction', desc: 'Binary classification to detect engines within 30 cycles of failure with 0.998 ROC-AUC.' },
  { icon: Zap,       title: 'Anomaly Detection',       desc: 'Isolation Forest identifies abnormal sensor patterns before they manifest as obvious RUL decline.' },
  { icon: Radio,     title: 'Sensor Intelligence',     desc: 'Explainable AI reveals which sensor channels drive predictions, with per-engine local analysis.' },
  { icon: Wrench,    title: 'Maintenance Planning',    desc: 'Deterministic rule engine converts prediction outputs into prioritised maintenance recommendations.' },
]

function AlertIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" {...props}>
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  )
}

export default function Landing() {
  const nav = useNavigate()

  return (
    <div className="min-h-screen bg-aero text-text-pri font-body overflow-x-hidden">
      {/* ── Hero ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-6">
        {/* Orbital rings */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden">
          {[400, 550, 700].map((size, i) => (
            <motion.div
              key={size}
              style={{ width: size, height: size }}
              animate={{ rotate: 360 }}
              transition={{ duration: 20 + i * 8, repeat: Infinity, ease: 'linear' }}
              className="absolute rounded-full border border-primary/5"
            />
          ))}
          <div className="absolute w-64 h-64 rounded-full bg-primary/5 blur-3xl" />
          <div className="absolute w-96 h-96 rounded-full bg-secondary/4 blur-3xl" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 max-w-4xl"
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/30 bg-primary/5 mb-8"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            <span className="text-xs font-mono text-primary tracking-[0.2em]">ADVANCED ENGINE INTELLIGENCE SYSTEM</span>
          </motion.div>

          <h1 className="text-6xl md:text-8xl font-display font-black tracking-tight mb-6 leading-none">
            <span className="text-text-pri">PREDICT</span>
            <br />
            <span className="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
              BEFORE FAILURE.
            </span>
          </h1>

          <p className="text-text-sec text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
            Transform raw engine telemetry into predictive intelligence. Detect abnormal degradation,
            estimate remaining useful life, understand failure risks, and make smarter maintenance
            decisions before equipment failure.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => nav('/app/overview')}
              className="btn-primary flex items-center gap-2 text-sm tracking-wider"
            >
              INITIALIZE MISSION CONTROL
              <ArrowRight size={16} />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => document.getElementById('tech')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn-ghost flex items-center gap-2 text-sm tracking-wider"
            >
              EXPLORE TECHNOLOGY
            </motion.button>
          </div>
        </motion.div>

        {/* Scroll cue */}
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 text-text-sec/40"
        >
          <ChevronDown size={20} />
        </motion.div>

        {/* Ticker strip */}
        <div className="absolute bottom-0 left-0 right-0 border-t border-border bg-bg-sec/50 backdrop-blur py-2 overflow-hidden">
          <div className="flex gap-12 px-6 text-[10px] font-mono text-text-sec/60 tracking-widest">
            {TICKER.map((t, i) => (
              <span key={i} className="whitespace-nowrap shrink-0">{t}</span>
            ))}
            {TICKER.map((t, i) => (
              <span key={`r${i}`} className="whitespace-nowrap shrink-0">{t}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── About ── */}
      <section className="py-24 px-6 max-w-5xl mx-auto" id="tech">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <p className="text-xs font-mono text-primary tracking-[0.2em] mb-4">ABOUT AEROSENTINEL</p>
          <h2 className="text-4xl font-display font-bold text-text-pri mb-4">Engine Intelligence Platform</h2>
          <p className="text-text-sec max-w-3xl mx-auto">
            AeroSentinel is an AI-powered predictive maintenance platform that analyses multivariate engine
            sensor telemetry to estimate Remaining Useful Life, detect abnormal degradation, identify influential
            sensor measurements, and recommend preventive maintenance actions. Trained and evaluated on the
            NASA C-MAPSS Turbofan Engine Degradation Dataset.
          </p>
        </motion.div>

        {/* Five capabilities */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              className="glass p-6 hover:-translate-y-1 transition-transform"
            >
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                <f.icon size={18} className="text-primary" />
              </div>
              <h3 className="font-display font-semibold text-text-pri mb-2">{f.title}</h3>
              <p className="text-sm text-text-sec leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Metrics ── */}
      <section className="py-20 px-6 bg-bg-sec/30">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <p className="text-xs font-mono text-accent tracking-[0.2em] mb-4">VERIFIED PERFORMANCE — FD001</p>
            <h2 className="text-4xl font-display font-bold mb-12">Model Performance</h2>
          </motion.div>
          <div className="grid md:grid-cols-2 gap-6">
            {/* RUL */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass p-6 text-left"
            >
              <p className="text-xs font-mono text-primary mb-4 tracking-wider">RUL REGRESSION — RANDOM FOREST</p>
              <div className="space-y-3">
                {[
                  { label: 'MAE', value: '13.15 cycles' },
                  { label: 'RMSE', value: '18.40' },
                  { label: 'R²', value: '0.804' },
                ].map(m => (
                  <div key={m.label} className="flex justify-between items-center border-b border-border pb-2">
                    <span className="text-sm text-text-sec font-mono">{m.label}</span>
                    <span className="text-sm font-bold text-primary font-mono">{m.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>
            {/* Failure */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass p-6 text-left"
            >
              <p className="text-xs font-mono text-accent mb-4 tracking-wider">FAILURE CLASSIFIER — GRADIENT BOOSTING</p>
              <div className="space-y-3">
                {[
                  { label: 'F1 Score', value: '0.943' },
                  { label: 'ROC-AUC', value: '0.998' },
                  { label: 'Recall', value: '1.000' },
                ].map(m => (
                  <div key={m.label} className="flex justify-between items-center border-b border-border pb-2">
                    <span className="text-sm text-text-sec font-mono">{m.label}</span>
                    <span className="text-sm font-bold text-accent font-mono">{m.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
          <p className="text-xs text-text-sec/50 font-mono mt-4">Source: FD001 Existing Training Run — evaluation on 100 test engines</p>
        </div>
      </section>

      {/* ── Dataset overview ── */}
      <section className="py-20 px-6 max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-12">
          <p className="text-xs font-mono text-secondary tracking-[0.2em] mb-4">TRAINING DATA</p>
          <h2 className="text-4xl font-display font-bold">NASA C-MAPSS Dataset</h2>
          <p className="text-text-sec mt-3 max-w-2xl mx-auto">
            The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset provides
            run-to-failure trajectories for turbofan engines across four subsets of varying complexity.
          </p>
        </motion.div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { ds: 'FD001', cond: 1, fault: 1, eng: 100,  c: 'Simple'  },
            { ds: 'FD002', cond: 6, fault: 1, eng: 260,  c: 'Medium'  },
            { ds: 'FD003', cond: 1, fault: 2, eng: 100,  c: 'Medium'  },
            { ds: 'FD004', cond: 6, fault: 2, eng: 249,  c: 'Complex' },
          ].map((d, i) => (
            <motion.div
              key={d.ds}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.07 }}
              className="glass-sm p-4 text-center"
            >
              <p className="font-mono font-bold text-primary text-lg">{d.ds}</p>
              <p className="text-xs text-text-sec mt-1">{d.eng} engines</p>
              <p className="text-xs text-text-sec">{d.cond} cond • {d.fault} fault</p>
              <span className="inline-block mt-2 text-[10px] font-mono px-2 py-0.5 rounded-full bg-secondary/10 text-secondary border border-secondary/20">{d.c}</span>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-24 px-6 text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <h2 className="text-5xl font-display font-black mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Ready to Monitor?
          </h2>
          <p className="text-text-sec text-lg mb-10 max-w-xl mx-auto">
            Enter AeroSentinel Mission Control and start analysing engine health with real ML predictions.
          </p>
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => nav('/app/overview')}
            className="btn-primary flex items-center gap-2 mx-auto text-sm tracking-wider"
          >
            LAUNCH MISSION CONTROL
            <ArrowRight size={16} />
          </motion.button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-6 px-6 flex flex-col md:flex-row justify-between items-center text-xs text-text-sec/50 font-mono gap-2">
        <span>AEROSENTINEL — INTELLIGENT ENGINE HEALTH MONITORING PLATFORM</span>
        <span>Trained on NASA C-MAPSS Dataset · Research Grade · Not for Aviation Use</span>
      </footer>
    </div>
  )
}
