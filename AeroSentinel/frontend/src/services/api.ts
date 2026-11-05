import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (r) => r,
  (err) => {
    console.error('API error:', err.response?.data || err.message)
    return Promise.reject(err)
  }
)

export default api

// ── System ──────────────────────────────────────────────────────────────────
export const getHealth      = ()       => api.get('/health')
export const getSystemStatus = ()      => api.get('/system/status')

// ── Dashboard ────────────────────────────────────────────────────────────────
export const getDashboardSummary = (dataset = 'FD001') =>
  api.get(`/dashboard/summary?dataset=${dataset}`, { timeout: 120000 })
export const getFleet            = (dataset = 'FD001') =>
  api.get(`/dashboard/fleet?dataset=${dataset}`,   { timeout: 120000 })
export const getFD001Metrics     = ()                  => api.get('/dashboard/metrics/fd001')

// ── Datasets ─────────────────────────────────────────────────────────────────
export const getDatasets        = ()                              => api.get('/datasets')
export const getDatasetSummary  = (subset: string)               => api.get(`/datasets/${subset}/summary`)
export const getEngines         = (subset: string, source = 'test') => api.get(`/datasets/${subset}/engines?source=${source}`)
export const getEngineSensors   = (subset: string, id: number)   => api.get(`/datasets/${subset}/engines/${id}/sensors`)
export const getEngineSnapshot  = (subset: string, id: number)   => api.get(`/datasets/${subset}/engines/${id}/snapshot`)
export const getLifecycleStats  = (subset: string)               => api.get(`/datasets/${subset}/lifecycle`)
export const getRULDistribution = (subset: string)               => api.get(`/datasets/${subset}/rul-distribution`)
export const getSensorCorrelations = (subset: string)            => api.get(`/datasets/${subset}/correlations`)

// ── Predictions ───────────────────────────────────────────────────────────────
export const predictRUL = (dataset: string, engine_unit: number, model_key = 'rul_rf') =>
  api.post('/predict/rul', { dataset, engine_unit, model_key })
export const predictFailure = (dataset: string, engine_unit: number, model_key = 'fail_gb') =>
  api.post('/predict/failure', { dataset, engine_unit, model_key })
export const getPredictionHistory = (limit = 50) => api.get(`/predict/history?limit=${limit}`)

// ── Anomaly ───────────────────────────────────────────────────────────────────
export const detectAnomaly     = (dataset: string, engine_unit: number) =>
  api.post('/anomaly/detect', { dataset, engine_unit })
export const getAnomalySummary = (dataset = 'FD001') => api.get(`/anomaly/summary?dataset=${dataset}`)
export const getAnomalyHistory = (limit = 50)        => api.get(`/anomaly/history?limit=${limit}`)

// ── Explanations ──────────────────────────────────────────────────────────────
export const getFeatureImportance  = (model_key = 'rul_rf', top_n = 30) =>
  api.get(`/explanations/feature-importance?model_key=${model_key}&top_n=${top_n}`)
export const getEngineExplanation  = (dataset: string, engine_unit: number, model_key = 'rul_rf') =>
  api.get(`/explanations/engine?dataset=${dataset}&engine_unit=${engine_unit}&model_key=${model_key}`)

// ── Maintenance ───────────────────────────────────────────────────────────────
export const generateRecommendation = (dataset: string, engine_unit: number) =>
  api.post('/maintenance/recommendations', { dataset, engine_unit })
export const getMaintenanceHistory  = (limit = 100) => api.get(`/maintenance/history?limit=${limit}`)
export const updateMaintenanceStatus = (id: number, status: string) =>
  api.patch(`/maintenance/${id}/status`, { status })

// ── Models ────────────────────────────────────────────────────────────────────
export const getModels          = () => api.get('/models')
export const getModelComparison = () => api.get('/models/comparison')

// ── Reports ───────────────────────────────────────────────────────────────────
export const getCSVReport = (dataset: string, engine_unit: number) =>
  `/api/reports/engine/csv?dataset=${dataset}&engine_unit=${engine_unit}`
export const getPDFReport = (dataset: string, engine_unit: number) =>
  `/api/reports/engine/pdf?dataset=${dataset}&engine_unit=${engine_unit}`
