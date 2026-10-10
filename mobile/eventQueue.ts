import AsyncStorage from "@react-native-async-storage/async-storage";

import type { ImpactPriority, ImuFeatures } from "./inference";

export type QueuedLiveEvent = ImuFeatures & {
  id: string;
  createdAt: string;
  event_type: "Road anomaly";
  latitude: number;
  longitude: number;
  image_base64: string;
  vision_probability: number;
  imu_probability: number;
  fused_probability: number;
  impact_priority: ImpactPriority;
};

const STORAGE_KEY = "geofusion.pending-live-events.v1";
const MAX_PENDING_EVENTS = 20;

async function read() {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  if (!stored) return [] as QueuedLiveEvent[];
  const events = JSON.parse(stored) as QueuedLiveEvent[];
  return Array.isArray(events) ? events : [];
}

export async function pendingEvents() {
  return read();
}

export async function queueEvent(event: QueuedLiveEvent) {
  const events = [...await read(), event].slice(-MAX_PENDING_EVENTS);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(events));
  return events;
}

export async function removeQueuedEvent(id: string) {
  const events = (await read()).filter((event) => event.id !== id);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(events));
  return events;
}
