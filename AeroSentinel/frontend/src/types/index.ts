export type HealthStatus = 'healthy' | 'warning' | 'high_risk' | 'critical'
export type RiskLevel = 'low' | 'medium' | 'high'
export type AnomalySeverity = 'normal' | 'unusual' | 'high_anomaly'
export type MaintenancePriority = 'normal' | 'medium' | 'high' | 'critical'
export type MaintenanceStatus = 'pending' | 'reviewed' | 'resolved'

export interface SystemStatus {
  backend: string
  models_ready: boolean
  datasets_available: string[]
  model_status: Record<string, boolean | number>
  db_connected: boolean
}

export interface DashboardSummary {
  dataset: string
  total_engines: number
  avg_predicted_rul: number
  engines_rul_below_30: number
  engines_rul_below_15: number
  engines_high_failure_risk: number
  critical_engines: number
  high_risk_engines: number
  warning_engines: number
  healthy_engines: number
  rul_mae: number
  rul_rmse: number
  rul_r2: number
  failure_f1: number
  failure_roc_auc: number
  source_label: string
}

export interface FleetEngine {
  engine_unit: number
  last_cycle: number
  predicted_rul: number | null
  true_rul: number | null
  health_status: HealthStatus
  failure_probability: number | null
  risk_level: RiskLevel | null
  anomaly_severity: AnomalySeverity | null
}

export interface RULPrediction {
  engine_unit: number
  dataset: string
  last_cycle: number
  predicted_rul: number
  true_rul: number | null
  model: string
  health_status: HealthStatus
  source: string
  prediction_id?: number
  timestamp?: string
}

export interface FailurePrediction {
  engine_unit: number
  dataset: string
  last_cycle: number
  failure_probability: number
  failure_predicted: boolean
  risk_level: RiskLevel
  model: string
  prediction_id?: number
  timestamp?: string
}

export interface AnomalyResult {
  engine_unit: number
  dataset: string
  overall_score: number
  severity: AnomalySeverity
  anomaly_count: number
  total_cycles: number
  cycles: CycleAnomaly[]
}

export interface CycleAnomaly {
  cycle: number
  anomaly_score: number
  is_anomaly: boolean
  severity: AnomalySeverity
  top_sensors: string[]
}

export interface FeatureImportanceItem {
  feature: string
  importance: number
  rank: number
}

export interface SensorHistory {
  sensor: string
  cycles: number[]
  values: number[]
}

export interface SensorSnapshot {
  sensor: string
  latest_value: number
  mean_value: number
  min_value: number
  max_value: number
}

export interface MaintenanceRecommendation {
  id: number
  engine_unit: number
  dataset_subset: string
  recommendation: string
  priority: MaintenancePriority
  evidence: Record<string, number | null>
  status: MaintenanceStatus
  created_at: string
  updated_at?: string
}

export interface ModelInfo {
  model_key: string
  display_name: string
  model_type: string
  dataset: string
  is_available: boolean
  metrics: Record<string, number>
}

export interface DatasetSummary {
  subset: string
  conditions: number
  fault_modes: number
  train_engines: number
  test_engines: number
  train_observations: number
  test_observations: number
  sensor_count: number
  complexity: string
  models_available: boolean
}
