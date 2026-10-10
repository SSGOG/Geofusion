import { useCallback, useEffect, useRef, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import WebView from "react-native-webview";

export type EventFeature = {
  id: string;
  geometry: { coordinates: [number, number] };
  properties: { event_type: string; impact_priority: string; fused_probability: number | null };
};

export type SegmentFeature = {
  id: number;
  geometry: { coordinates: [number, number][] };
  properties: { maintenance_priority: "high" | "medium" | "low"; event_count: number };
};

export type PhoneLocation = { latitude: number; longitude: number };

const MAP_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>GeoFusion OpenStreetMap</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map { width: 100%; height: 100%; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; }
    #map-status { position: absolute; z-index: 1000; top: 45%; width: 100%; color: #334e68; text-align: center; }
    .leaflet-container { background: #e7edf2; font-family: system-ui, sans-serif; }
  </style>
</head>
<body>
  <div id="map" role="application" aria-label="OpenStreetMap with GeoFusion road events"></div>
  <div id="map-status" role="status">Loading OpenStreetMap…</div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    (function () {
      var status = document.getElementById("map-status");
      var sendMessage = function (message) {
        if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage(message);
      };
      if (!window.L) {
        status.textContent = "Map renderer unavailable. Check your mobile internet connection.";
        sendMessage("renderer-error");
        return;
      }

      var map = L.map("map", { zoomControl: true, keyboard: true }).setView([0, 0], 2);
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(map);
      var overlays = L.layerGroup().addTo(map);
      var centered = false;

      var colorFor = function (priority) {
        if (priority === "high") return "#b71c1c";
        if (priority === "medium") return "#ef6c00";
        if (priority === "low") return "#1565c0";
        return "#616161";
      };
      var validPosition = function (latitude, longitude) {
        return Number.isFinite(latitude) && Number.isFinite(longitude)
          && Math.abs(latitude) <= 90 && Math.abs(longitude) <= 180;
      };
      var addPopup = function (marker, titleText, detailText) {
        var popup = document.createElement("div");
        var title = document.createElement("strong");
        title.textContent = titleText;
        var detail = document.createElement("div");
        detail.textContent = detailText;
        popup.appendChild(title);
        popup.appendChild(detail);
        marker.bindPopup(popup);
      };

      window.updateGeoFusionMap = function (data) {
        overlays.clearLayers();
        var center = data.phoneLocation
          ? [data.phoneLocation.latitude, data.phoneLocation.longitude]
          : null;

        data.segments.forEach(function (segment) {
          var positions = segment.geometry.coordinates;
          if (positions.length < 2 || positions.some(function (position) {
            return !validPosition(position[1], position[0]);
          })) return;
          L.polyline(positions.map(function (position) {
            return [position[1], position[0]];
          }), {
            color: colorFor(segment.properties.maintenance_priority),
            weight: 4,
            opacity: 0.85
          }).addTo(overlays);
        });

        data.events.forEach(function (event) {
          var longitude = event.geometry.coordinates[0];
          var latitude = event.geometry.coordinates[1];
          if (!validPosition(latitude, longitude)) return;
          var marker = L.circleMarker([latitude, longitude], {
            radius: 8,
            color: "#ffffff",
            weight: 2,
            fillColor: colorFor(event.properties.impact_priority),
            fillOpacity: 0.95
          }).addTo(overlays);
          var probability = Number.isFinite(event.properties.fused_probability)
            ? event.properties.fused_probability.toFixed(2)
            : "unavailable";
          addPopup(marker, event.properties.event_type, event.properties.impact_priority
            + " impact · fusion " + probability);
          if (!center) center = [latitude, longitude];
        });

        if (data.phoneLocation && validPosition(data.phoneLocation.latitude, data.phoneLocation.longitude)) {
          var location = L.circleMarker([data.phoneLocation.latitude, data.phoneLocation.longitude], {
            radius: 7,
            color: "#ffffff",
            weight: 2,
            fillColor: "#1565c0",
            fillOpacity: 1
          }).addTo(overlays);
          addPopup(location, "Current phone location", "GPS position");
        }

        if (!centered && center && validPosition(center[0], center[1])) {
          map.setView(center, 15);
          centered = true;
        }
      };

      status.remove();
      sendMessage("ready");
    })();
  </script>
</body>
</html>`;

function toScriptValue(value: unknown) {
  const serialized = JSON.stringify(value);
  if (!serialized) throw new Error("Unable to serialize OpenStreetMap features.");
  return serialized.replace(/[<>&\u2028\u2029]/g, (character) =>
    `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`);
}

type OpenStreetMapProps = {
  events: EventFeature[];
  segments: SegmentFeature[];
  phoneLocation: PhoneLocation | null;
};

export default function OpenStreetMap({ events, segments, phoneLocation }: OpenStreetMapProps) {
  const webView = useRef<WebView>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  const handleMessage = useCallback((message: string) => {
    setStatus(message === "ready" ? "ready" : "error");
  }, []);

  useEffect(() => {
    if (status !== "ready") return;
    const data = toScriptValue({ events, segments, phoneLocation });
    webView.current?.injectJavaScript(
      `(function () {
        var payload = ${data};
        var attempts = 0;
        var apply = function () {
          if (typeof window.updateGeoFusionMap === "function") {
            window.updateGeoFusionMap(payload);
            return;
          }
          attempts += 1;
          if (attempts < 20) setTimeout(apply, 100);
          else if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage("renderer-error");
        };
        apply();
      })(); true;`,
    );
  }, [events, phoneLocation, segments, status]);

  return (
    <View style={styles.container}>
      <WebView
        ref={webView}
        accessibilityLabel="OpenStreetMap with GeoFusion road events"
        originWhitelist={["https://*"]}
        onShouldStartLoadWithRequest={(request) => request.url.startsWith("https://geofusion.local/")}
        source={{ html: MAP_HTML, baseUrl: "https://geofusion.local/" }}
        javaScriptEnabled
        domStorageEnabled={false}
        mixedContentMode="never"
        setSupportMultipleWindows={false}
        onMessage={(event) => handleMessage(event.nativeEvent.data)}
        onError={() => setStatus("error")}
        style={styles.webView}
      />
      {status !== "ready" && (
        <View pointerEvents="none" style={styles.status}>
          {status === "loading" && <ActivityIndicator color="#1b5e20" />}
          <Text style={styles.statusText}>
            {status === "loading" ? "Loading OpenStreetMap…" : "Map unavailable. Check your mobile internet connection."}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  webView: { backgroundColor: "#e7edf2", flex: 1 },
  status: {
    alignItems: "center",
    backgroundColor: "#e7edf2",
    bottom: 0,
    gap: 8,
    justifyContent: "center",
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
  statusText: { color: "#334e68", fontSize: 13, textAlign: "center" },
});
