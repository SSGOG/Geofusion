import { useCallback, useEffect, useRef, useState } from "react";
import { ActivityIndicator, Alert, Platform, Pressable, StatusBar, StyleSheet, Text, View } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as Location from "expo-location";
import { Accelerometer, Gyroscope } from "expo-sensors";

import { pendingEvents, queueEvent, QueuedLiveEvent, removeQueuedEvent } from "./eventQueue";
import { impactPriority, ImuFeatures, scoreLateFusion } from "./inference";
import OpenStreetMap, { EventFeature, PhoneLocation, SegmentFeature } from "./OpenStreetMap";
import { cloudSyncEnabled, dismissCloudEvent, listCloudEvents, syncCloudEvent } from "./supabaseSync";

type FeatureCollection<T> = { features: T[] };
type TimedValue = { timestamp: number; value: number };

const API_URL = process.env.EXPO_PUBLIC_GEOFUSION_API_URL;
const SENSOR_WINDOW_MS = 5_000;
const AUTO_CAPTURE_INTERVAL_MS = 3_000;
const IMPACT_TRIGGER_MPS2 = 1.5;

function liveEventFeature(event: QueuedLiveEvent): EventFeature {
  return {
    id: event.id,
    geometry: { coordinates: [event.longitude, event.latitude] },
    properties: {
      event_type: event.event_type,
      impact_priority: event.impact_priority,
      fused_probability: event.fused_probability,
    },
  };
}

function magnitude(x: number, y: number, z: number) {
  return Math.sqrt(x ** 2 + y ** 2 + z ** 2);
}

function appendSample(samples: TimedValue[], value: number) {
  const timestamp = Date.now();
  samples.push({ timestamp, value });
  while (samples[0]?.timestamp < timestamp - SENSOR_WINDOW_MS) samples.shift();
}

function errorDetail(detail: unknown) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => typeof item === "object" && item && "msg" in item ? String(item.msg) : "Invalid request").join("; ");
  return "Live event scoring failed.";
}

async function postLiveEvent(event: QueuedLiveEvent) {
  if (!API_URL) throw new Error("No map synchronisation URL is configured.");
  const response = await fetch(`${API_URL}/live-events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...event, client_event_id: event.id }),
  });
  const result = await response.json() as { mapped: boolean; event_id?: string; impact_priority?: string; fused_probability: number; detail?: unknown };
  if (!response.ok) throw new Error(errorDetail(result.detail));
  return result;
}

function hasDeviceScores(event: QueuedLiveEvent) {
  return Number.isFinite(event.vision_probability) && Number.isFinite(event.imu_probability) && Number.isFinite(event.fused_probability);
}

async function synchroniseEvent(event: QueuedLiveEvent) {
  if (cloudSyncEnabled && hasDeviceScores(event)) return syncCloudEvent(event);
  return postLiveEvent(event);
}

function sensorFeatures(acceleration: TimedValue[], gyroscope: TimedValue[]): ImuFeatures | null {
  if (acceleration.length < 5 || gyroscope.length < 5) return null;
  const baseline = acceleration.reduce((sum, sample) => sum + sample.value, 0) / acceleration.length;
  const deviations = acceleration.map((sample) => sample.value - baseline);
  const rms = Math.sqrt(deviations.reduce((sum, value) => sum + value ** 2, 0) / deviations.length);
  const peak = Math.max(...deviations.map((value) => Math.abs(value)));
  const jerk = deviations.slice(1).map((value, index) => (value - deviations[index]) / Math.max((acceleration[index + 1].timestamp - acceleration[index].timestamp) / 1000, 0.001));
  const jerkRms = Math.sqrt(jerk.reduce((sum, value) => sum + value ** 2, 0) / jerk.length);
  const gyroPeak = Math.max(...gyroscope.map((sample) => sample.value));
  const duration = (acceleration[acceleration.length - 1].timestamp - acceleration[0].timestamp) / 1000;
  const samplingHz = acceleration.length / Math.max(duration, 0.001);
  return { imu_sampling_hz: samplingHz, linear_acc_rms_deviation_mps2: rms, linear_acc_peak_deviation_mps2: peak, linear_acc_jerk_rms_mps3: jerkRms, gyro_peak_radps: gyroPeak };
}

function sensorSummary(acceleration: TimedValue[], gyroscope: TimedValue[]) {
  const values = sensorFeatures(acceleration, gyroscope);
  if (!values) return "Too few sensor samples were captured.";
  return `${acceleration.length} accelerometer and ${gyroscope.length} gyroscope samples; ${values.imu_sampling_hz.toFixed(1)} Hz; RMS ${values.linear_acc_rms_deviation_mps2.toFixed(2)} m/s²; peak ${values.linear_acc_peak_deviation_mps2.toFixed(2)} m/s²; jerk RMS ${values.linear_acc_jerk_rms_mps3.toFixed(2)} m/s³; gyro peak ${values.gyro_peak_radps.toFixed(2)} rad/s.`;
}

export default function App() {
  const [events, setEvents] = useState<EventFeature[]>([]);
  const [segments, setSegments] = useState<SegmentFeature[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("Loading mapped GeoFusion results…");
  const [capturing, setCapturing] = useState(false);
  const [sensorMessage, setSensorMessage] = useState("Sensor capture is idle.");
  const [phoneLocation, setPhoneLocation] = useState<PhoneLocation | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [autoDetecting, setAutoDetecting] = useState(false);
  const [lastLiveEventId, setLastLiveEventId] = useState<string | null>(null);
  const [lastQueuedEventId, setLastQueuedEventId] = useState<string | null>(null);
  const [queuedCount, setQueuedCount] = useState(0);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const acceleration = useRef<TimedValue[]>([]);
  const gyroscope = useRef<TimedValue[]>([]);
  const accelerationSubscription = useRef<ReturnType<typeof Accelerometer.addListener> | null>(null);
  const gyroscopeSubscription = useRef<ReturnType<typeof Gyroscope.addListener> | null>(null);
  const capturingRef = useRef(false);
  const camera = useRef<CameraView>(null);
  const autoCaptureInFlight = useRef(false);
  const autoCaptureInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const queueSyncInFlight = useRef(false);

  const stopSensorCapture = useCallback(() => {
    accelerationSubscription.current?.remove();
    gyroscopeSubscription.current?.remove();
    accelerationSubscription.current = null;
    gyroscopeSubscription.current = null;
    if (capturingRef.current) setSensorMessage(sensorSummary(acceleration.current, gyroscope.current));
    capturingRef.current = false;
    setCapturing(false);
  }, []);

  const startSensorCapture = useCallback(() => {
    acceleration.current = [];
    gyroscope.current = [];
    Accelerometer.setUpdateInterval(20);
    Gyroscope.setUpdateInterval(20);
    accelerationSubscription.current = Accelerometer.addListener(({ x, y, z }) => appendSample(acceleration.current, magnitude(x, y, z) * 9.80665));
    gyroscopeSubscription.current = Gyroscope.addListener(({ x, y, z }) => appendSample(gyroscope.current, magnitude(x, y, z)));
    capturingRef.current = true;
    setSensorMessage("Collecting phone accelerometer and gyroscope readings at a 50 Hz target.");
    setCapturing(true);
  }, []);

  const openCamera = useCallback(async () => {
    if (!cameraPermission?.granted) {
      const permission = await requestCameraPermission();
      if (!permission.granted) {
        Alert.alert("Camera permission required", "Allow camera access to collect road-surface evidence.");
        return false;
      }
    }
    setCameraOpen(true);
    return true;
  }, [cameraPermission?.granted, requestCameraPermission]);

  const syncPendingEvents = useCallback(async () => {
    if ((!API_URL && !cloudSyncEnabled) || queueSyncInFlight.current) return;
    queueSyncInFlight.current = true;
    try {
      for (const event of await pendingEvents()) {
        try {
          const result = await synchroniseEvent(event);
          await removeQueuedEvent(event.id);
          if (result.mapped && result.event_id) {
            setLastLiveEventId(result.event_id);
            setEvents((current) => [...current.filter((item) => item.id !== event.id), liveEventFeature(event)]);
          }
        } catch {
          break;
        }
      }
      const remaining = await pendingEvents();
      setQueuedCount(remaining.length);
      setLastQueuedEventId(remaining.at(-1)?.id ?? null);
    } finally {
      queueSyncInFlight.current = false;
    }
  }, []);

  const captureLiveEvent = useCallback(async () => {
    const features = sensorFeatures(acceleration.current, gyroscope.current);
    if (!features || features.linear_acc_peak_deviation_mps2 < IMPACT_TRIGGER_MPS2 || !camera.current || autoCaptureInFlight.current) {
      return;
    }
    autoCaptureInFlight.current = true;
    try {
      const photo = await camera.current.takePictureAsync({ base64: true, quality: 0.15, skipProcessing: true });
      if (!photo?.base64) throw new Error("Camera did not return an image.");
      const localScore = await scoreLateFusion(photo.base64, features);
      setMessage(`On-device fusion: vision ${localScore.visionProbability.toFixed(2)}, IMU ${localScore.imuProbability.toFixed(2)}, combined ${localScore.fusedProbability.toFixed(2)}.`);
      if (!phoneLocation || localScore.fusedProbability < 0.5) return;
      const event: QueuedLiveEvent = { id: `phone_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, createdAt: new Date().toISOString(), event_type: "Road anomaly", latitude: phoneLocation.latitude, longitude: phoneLocation.longitude, image_base64: photo.base64, vision_probability: localScore.visionProbability, imu_probability: localScore.imuProbability, fused_probability: localScore.fusedProbability, impact_priority: impactPriority(features), ...features };
      try {
        const result = await synchroniseEvent(event);
        if (result.mapped && result.event_id) {
          setLastLiveEventId(result.event_id);
          setEvents((current) => [...current.filter((item) => item.id !== event.id), liveEventFeature(event)]);
          setMessage(`Road anomaly mapped with ${result.impact_priority} relative sensor impact (fusion ${result.fused_probability.toFixed(2)}).`);
        }
      } catch {
        const events = await queueEvent(event);
        setQueuedCount(events.length);
        setLastQueuedEventId(event.id);
        setMessage(`On-device candidate saved locally; ${events.length} event${events.length === 1 ? "" : "s"} waiting for synchronisation.`);
      }
    } catch (error) {
      setMessage(`Live capture failed: ${error instanceof Error ? error.message : "Unable to score the live event."}`);
    } finally {
      autoCaptureInFlight.current = false;
    }
  }, [phoneLocation]);

  const startLiveDetection = useCallback(async () => {
    if (!await openCamera()) return;
    startSensorCapture();
    setAutoDetecting(true);
    autoCaptureInterval.current = setInterval(() => void captureLiveEvent(), AUTO_CAPTURE_INTERVAL_MS);
    setMessage("Live detection is active: scanning IMU impact windows and road frames automatically.");
  }, [captureLiveEvent, openCamera, startSensorCapture]);

  const stopLiveDetection = useCallback(() => {
    if (autoCaptureInterval.current) clearInterval(autoCaptureInterval.current);
    autoCaptureInterval.current = null;
    stopSensorCapture();
    setCameraOpen(false);
    setAutoDetecting(false);
    setMessage("Live detection stopped.");
  }, [stopSensorCapture]);

  const dismissLastLiveEvent = useCallback(async () => {
    if (lastQueuedEventId) {
      const events = await removeQueuedEvent(lastQueuedEventId);
      setQueuedCount(events.length);
      setLastQueuedEventId(events.at(-1)?.id ?? null);
      setMessage("Latest unsynchronised event removed from this phone.");
      return;
    }
    if (!lastLiveEventId || (!API_URL && !cloudSyncEnabled)) return;
    try {
      if (cloudSyncEnabled) await dismissCloudEvent(lastLiveEventId);
      else {
        const response = await fetch(`${API_URL}/live-events/${lastLiveEventId}`, { method: "DELETE" });
        if (!response.ok) throw new Error("The mapped event could not be removed.");
      }
      setLastLiveEventId(null);
      setEvents((current) => current.filter((event) => event.id !== lastLiveEventId));
      setMessage("Latest live event removed from the map and retained as a dismissed audit record.");
    } catch (error) {
      Alert.alert("Removal failed", error instanceof Error ? error.message : "Unable to remove the live event.");
    }
  }, [lastLiveEventId, lastQueuedEventId]);

  const loadMapData = useCallback(async () => {
    if (!API_URL && !cloudSyncEnabled) {
      setLoading(false);
      setMessage("Set EXPO_PUBLIC_GEOFUSION_API_URL in .env before starting the app.");
      return;
    }
    setLoading(true);
    try {
      if (cloudSyncEnabled) {
        const cloudEvents = await listCloudEvents();
        setEvents(cloudEvents.map((event) => ({
          id: event.event_id,
          geometry: { coordinates: [event.longitude, event.latitude] },
          properties: { event_type: event.event_type, impact_priority: event.impact_priority, fused_probability: event.fused_probability },
        })));
        setSegments([]);
        await syncPendingEvents();
        setMessage(`${cloudEvents.length} active GeoFusion cloud event${cloudEvents.length === 1 ? "" : "s"} loaded.`);
        return;
      }
      const [eventsResponse, segmentsResponse] = await Promise.all([
        fetch(`${API_URL}/events.geojson`),
        fetch(`${API_URL}/segments.geojson`),
      ]);
      if (!eventsResponse.ok || !segmentsResponse.ok) throw new Error("The GeoFusion API did not return map data.");
      const eventData = (await eventsResponse.json()) as FeatureCollection<EventFeature>;
      const segmentData = (await segmentsResponse.json()) as FeatureCollection<SegmentFeature>;
      setEvents(eventData.features);
      setSegments(segmentData.features);
      await syncPendingEvents();
      setMessage(`${eventData.features.length} events and ${segmentData.features.length} prioritised road segments loaded.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load map data.");
    } finally {
      setLoading(false);
    }
  }, [syncPendingEvents]);

  useEffect(() => {
    void loadMapData();
  }, [loadMapData]);

  useEffect(() => {
    void pendingEvents().then((items) => {
      setQueuedCount(items.length);
      setLastQueuedEventId(items.at(-1)?.id ?? null);
      return syncPendingEvents();
    }).catch(() => setMessage("Saved live-event queue could not be read."));
  }, [syncPendingEvents]);

  useEffect(() => {
    void (async () => {
      const permission = await Location.requestForegroundPermissionsAsync();
      if (permission.status !== "granted") return;
      const current = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      setPhoneLocation({ latitude: current.coords.latitude, longitude: current.coords.longitude });
      setMessage((previous) => `${previous} Current GPS: ${current.coords.latitude.toFixed(5)}, ${current.coords.longitude.toFixed(5)}.`);
    })().catch(() => Alert.alert("Location unavailable", "The map still works without phone location permission."));
  }, []);

  useEffect(() => () => {
    accelerationSubscription.current?.remove();
    gyroscopeSubscription.current?.remove();
    if (autoCaptureInterval.current) clearInterval(autoCaptureInterval.current);
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View><Text style={styles.title}>GeoFusion</Text><Text style={styles.subtitle}>Road-condition intelligence map</Text></View>
        <Pressable style={styles.button} onPress={() => void loadMapData()}><Text style={styles.buttonText}>Refresh</Text></Pressable>
      </View>
      <View style={styles.map}>
        <OpenStreetMap events={events} segments={segments} phoneLocation={phoneLocation} />
        {cameraOpen && <CameraView ref={camera} style={styles.cameraOverlay} facing="back" />}
      </View>
      <View style={styles.captureControls}>
        <Pressable accessibilityLabel={autoDetecting ? "Stop live detection" : "Start live detection"} style={[styles.captureButton, autoDetecting && styles.stopButton]} onPress={() => void (autoDetecting ? stopLiveDetection() : startLiveDetection())}><Text style={styles.buttonText}>{autoDetecting ? "Stop live detection" : "Start live detection"}</Text></Pressable>
        {(lastLiveEventId || lastQueuedEventId) && <Pressable accessibilityLabel="Remove the latest road event" style={styles.removeButton} onPress={() => void dismissLastLiveEvent()}><Text style={styles.buttonText}>Remove latest</Text></Pressable>}
      </View>
      <View style={styles.footer}>
        {loading && <ActivityIndicator color="#1b5e20" />}
        <View style={styles.statusGroup}>
          <Text style={styles.status}>{message}{queuedCount ? ` ${queuedCount} event${queuedCount === 1 ? "" : "s"} awaiting sync.` : ""}</Text>
          <Text style={styles.sensorStatus}>{sensorMessage}</Text>
        </View>
        <Pressable style={[styles.sensorButton, capturing && styles.stopButton]} onPress={capturing ? stopSensorCapture : startSensorCapture}>
          <Text style={styles.buttonText}>{capturing ? "Stop sensing" : "Start sensing"}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#ffffff", paddingTop: Platform.OS === "android" ? StatusBar.currentHeight ?? 0 : 0 },
  header: { alignItems: "center", flexDirection: "row", justifyContent: "space-between", padding: 16 },
  title: { color: "#102a43", fontSize: 24, fontWeight: "700" },
  subtitle: { color: "#486581", marginTop: 2 },
  button: { backgroundColor: "#1b5e20", borderRadius: 6, paddingHorizontal: 14, paddingVertical: 9 },
  buttonText: { color: "#ffffff", fontWeight: "600" },
  map: { flex: 1 },
  cameraOverlay: { bottom: 0, left: 0, position: "absolute", right: 0, top: 0 },
  captureControls: { alignItems: "center", flexDirection: "row", flexWrap: "wrap", gap: 7, justifyContent: "center", padding: 8 },
  captureButton: { backgroundColor: "#00695c", borderRadius: 5, paddingHorizontal: 10, paddingVertical: 7 },
  removeButton: { backgroundColor: "#b71c1c", borderRadius: 5, paddingHorizontal: 10, paddingVertical: 7 },
  footer: { alignItems: "center", flexDirection: "row", gap: 10, minHeight: 100, paddingHorizontal: 16 },
  statusGroup: { flex: 1 },
  status: { color: "#243b53", flex: 1, fontSize: 12 },
  sensorStatus: { color: "#52606d", fontSize: 11, marginTop: 4 },
  sensorButton: { backgroundColor: "#0d47a1", borderRadius: 6, paddingHorizontal: 10, paddingVertical: 9 },
  stopButton: { backgroundColor: "#b71c1c" }
});
