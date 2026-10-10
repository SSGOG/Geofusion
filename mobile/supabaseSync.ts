import AsyncStorage from "@react-native-async-storage/async-storage";
import "react-native-url-polyfill/auto";
import { createClient } from "@supabase/supabase-js";

import { QueuedLiveEvent } from "./eventQueue";

const url = process.env.EXPO_PUBLIC_SUPABASE_URL;
const key = process.env.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
const supabase = url && key ? createClient(url, key, {
  auth: { storage: AsyncStorage, autoRefreshToken: true, persistSession: true, detectSessionInUrl: false },
}) : null;

export const cloudSyncEnabled = Boolean(supabase);

export type CloudRoadEvent = {
  event_id: string;
  event_type: string;
  latitude: number;
  longitude: number;
  fused_probability: number;
  impact_priority: string;
};

async function client() {
  if (!supabase) throw new Error("Supabase is not configured.");
  const { data: { session }, error: sessionError } = await supabase.auth.getSession();
  if (sessionError) throw sessionError;
  if (session) return supabase;
  const { error } = await supabase.auth.signInAnonymously();
  if (error) throw error;
  return supabase;
}

export async function listCloudEvents() {
  const active = await client();
  const { data, error } = await active.from("mobile_road_events")
    .select("event_id,event_type,latitude,longitude,fused_probability,impact_priority")
    .eq("status", "active")
    .order("created_at", { ascending: false })
    .limit(500);
  if (error) throw error;
  return (data ?? []) as CloudRoadEvent[];
}

export async function syncCloudEvent(event: QueuedLiveEvent) {
  const active = await client();
  const { error } = await active.from("mobile_road_events").upsert({
    event_id: event.id,
    event_type: event.event_type,
    latitude: event.latitude,
    longitude: event.longitude,
    vision_probability: event.vision_probability,
    imu_probability: event.imu_probability,
    fused_probability: event.fused_probability,
    impact_priority: event.impact_priority,
    created_at: event.createdAt,
  }, { onConflict: "event_id", ignoreDuplicates: true });
  if (error) throw error;
  return { mapped: true, event_id: event.id, impact_priority: event.impact_priority, fused_probability: event.fused_probability };
}

export async function dismissCloudEvent(eventId: string) {
  const active = await client();
  const { data, error } = await active.from("mobile_road_events")
    .update({ status: "dismissed", dismissed_at: new Date().toISOString() })
    .eq("event_id", eventId).eq("status", "active").select("event_id").maybeSingle();
  if (error) throw error;
  if (!data) throw new Error("The active cloud event was not found.");
}
