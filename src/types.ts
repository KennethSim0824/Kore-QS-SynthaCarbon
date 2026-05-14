export type VehicleClass = 'excavator' | 'tractor' | 'truck' | 'crane';

export interface VehicleDetection {
  class: VehicleClass;
  confidence: number;
  bbox: [number, number, number, number]; // [left%, top%, width%, height%]
}

export interface EmissionData {
  time: string;
  co2: number;
}

export interface AgentMessage {
  time: string;
  text: string;
}

export const HEAVY_DIESEL_CLASSES: VehicleClass[] = ['excavator', 'tractor', 'truck', 'crane'];

export const EMISSION_FACTORS: Record<VehicleClass, number> = {
  excavator: 12000,
  tractor: 9500,
  truck: 15000,
  crane: 5500,
};

export const VEHICLE_ICONS: Record<VehicleClass, string> = {
  excavator: '🏗️',
  tractor: '🚜',
  truck: '🚛',
  crane: '🏚️',
};

export const VEHICLE_COLORS: Record<VehicleClass, string> = {
  truck: '#ff3d57',
  excavator: '#ffb300',
  tractor: '#00e676',
  crane: '#448aff',
};
