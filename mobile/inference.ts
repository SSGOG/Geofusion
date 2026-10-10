import { Asset } from "expo-asset";
import { toByteArray } from "base64-js";
import { decode } from "jpeg-js";
import * as ort from "onnxruntime-react-native";

import model from "./assets/models/late_fusion_imu.json";
import impactReference from "./assets/models/impact_reference.json";

export type ImuFeatures = {
  imu_sampling_hz: number;
  linear_acc_rms_deviation_mps2: number;
  linear_acc_peak_deviation_mps2: number;
  linear_acc_jerk_rms_mps3: number;
  gyro_peak_radps: number;
};

type Score = { visionProbability: number; imuProbability: number; fusedProbability: number };
export type ImpactPriority = "low" | "medium" | "high";

const IMAGE_SIZE = 224;
let session: ort.InferenceSession | null = null;

function sigmoid(value: number) {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function pixel(data: Uint8Array, width: number, height: number, x: number, y: number, channel: number) {
  const safeX = Math.max(0, Math.min(width - 1, x));
  const safeY = Math.max(0, Math.min(height - 1, y));
  return data[(safeY * width + safeX) * 4 + channel];
}

function cameraTensor(base64: string) {
  const image = decode(toByteArray(base64), { useTArray: true, formatAsRGBA: true });
  if (!image.width || !image.height) throw new Error("Camera image could not be decoded.");
  const scale = IMAGE_SIZE / Math.min(image.width, image.height);
  const resizedWidth = Math.round(image.width * scale);
  const resizedHeight = Math.round(image.height * scale);
  const offsetX = (resizedWidth - IMAGE_SIZE) / 2;
  const offsetY = (resizedHeight - IMAGE_SIZE) / 2;
  const values = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);

  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    const sourceY = ((y + offsetY + 0.5) / scale) - 0.5;
    const y0 = Math.floor(sourceY);
    const yWeight = sourceY - y0;
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      const sourceX = ((x + offsetX + 0.5) / scale) - 0.5;
      const x0 = Math.floor(sourceX);
      const xWeight = sourceX - x0;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = pixel(image.data, image.width, image.height, x0, y0, channel) * (1 - xWeight) + pixel(image.data, image.width, image.height, x0 + 1, y0, channel) * xWeight;
        const bottom = pixel(image.data, image.width, image.height, x0, y0 + 1, channel) * (1 - xWeight) + pixel(image.data, image.width, image.height, x0 + 1, y0 + 1, channel) * xWeight;
        values[channel * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] = (top * (1 - yWeight) + bottom * yWeight) / 255;
      }
    }
  }
  return new ort.Tensor("float32", values, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

async function visualSession() {
  if (session) return session;
  const [asset] = await Asset.loadAsync(require("./assets/models/geofusion_yolo_late.onnx"));
  if (!asset.localUri) throw new Error("The visual model could not be loaded on this device.");
  session = await ort.InferenceSession.create(asset.localUri);
  return session;
}

function imuProbability(features: ImuFeatures) {
  const imu = model.imu;
  let logit = imu.positive_class_intercept;
  for (let index = 0; index < imu.feature_order.length; index += 1) {
    const feature = imu.feature_order[index] as keyof ImuFeatures;
    // shortcut: sampling rate was constant in training; replace it until phone-specific calibration is available.
    const value = feature === "imu_sampling_hz" ? imu.input_overrides.imu_sampling_hz : features[feature];
    const input = Number.isFinite(value) ? value : imu.imputer_median[index];
    logit += imu.positive_class_coefficients[index] * ((input - imu.scaler_mean[index]) / imu.scaler_scale[index]);
  }
  return sigmoid(logit);
}

export async function scoreLateFusion(base64: string, features: ImuFeatures): Promise<Score> {
  const activeSession = await visualSession();
  const outputs = await activeSession.run({ [activeSession.inputNames[0]]: cameraTensor(base64) });
  const probabilities = outputs[activeSession.outputNames[0]].data as Float32Array;
  const visionProbability = probabilities[0]; // class 0 is annotated_event in the selected training export.
  const imu = imuProbability(features);
  return { visionProbability, imuProbability: imu, fusedProbability: (visionProbability + imu) / 2 };
}

export function impactPriority(features: ImuFeatures): ImpactPriority {
  const score = Object.entries(impactReference.weights).reduce((sum, [feature, weight]) => {
    const values = impactReference.reference[feature as keyof typeof impactReference.reference];
    const value = features[feature as keyof ImuFeatures];
    let lower = 0;
    let upper = values.length;
    while (lower < upper) {
      const middle = Math.floor((lower + upper) / 2);
      if (values[middle] <= value) lower = middle + 1;
      else upper = middle;
    }
    return sum + weight * lower / values.length;
  }, 0);
  return score >= 2 / 3 ? "high" : score >= 1 / 3 ? "medium" : "low";
}
