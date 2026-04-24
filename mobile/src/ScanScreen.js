import { useNavigation } from "@react-navigation/native";
import axios from "axios";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import * as ImagePicker from "expo-image-picker";
import * as Linking from "expo-linking";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Animated,
  StyleSheet,
  Text,
  TouchableOpacity,
  Vibration,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { API_URL } from "./config";

const VIDEO_DURATION_MS = 1500;  // kısa video süresi — exposure ayarlanır

export default function ScanScreen() {
  const navigation = useNavigation();
  const [permission, requestPermission] = useCameraPermissions();
  const [status, setStatus] = useState("scanning");
  const [foundLabel, setFoundLabel] = useState(null);
  const [foundUrl, setFoundUrl] = useState(null);

  const cameraRef = useRef(null);
  const scanning = useRef(false);
  const intervalRef = useRef(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const dotAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.04, duration: 900, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
      ])
    ).start();
    Animated.loop(
      Animated.sequence([
        Animated.timing(dotAnim, { toValue: 0.3, duration: 600, useNativeDriver: true }),
        Animated.timing(dotAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  const showFound = useCallback((url, label) => {
    Vibration.vibrate([0, 80, 60, 80]);
    clearInterval(intervalRef.current);
    scanning.current = false;
    setFoundUrl(url);
    setFoundLabel(label || url);
    setStatus("found");
  }, []);

  const resetToScanning = useCallback(() => {
    setStatus("scanning");
    setFoundUrl(null);
    setFoundLabel(null);
    scanning.current = false;
  }, []);

  const sendSingleFrame = useCallback(async (uri) => {
    const form = new FormData();
    form.append("image", { uri, type: "image/jpeg", name: "frame.jpg" });
    const res = await axios.post(`${API_URL}/api/ss/decode`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 20000,
    });
    return res.data;
  }, []);

  const sendVideo = useCallback(async (uri) => {
    const form = new FormData();
    form.append("video", { uri, type: "video/mp4", name: "scan.mp4" });
    const res = await axios.post(`${API_URL}/api/decode-video`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 30000,
    });
    return res.data;
  }, []);

  const scanFrame = useCallback(async () => {
    if (scanning.current || !cameraRef.current) return;
    scanning.current = true;
    setStatus("processing");
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.9, skipProcessing: true });
      const data = await sendSingleFrame(photo.uri);
      showFound(data.url, data.label);
    } catch {
      scanning.current = false;
      setStatus("scanning");
      intervalRef.current = setTimeout(scanFrame, 2000);
    }
  }, [sendSingleFrame, showFound]);

  useEffect(() => {
    if (!permission?.granted) return;
    // Kamera sensörü hazır olana kadar 2sn bekle, sonra video scan başlat
    const startup = setTimeout(() => scanFrame(), 2000);
    return () => {
      clearTimeout(startup);
      clearTimeout(intervalRef.current);
      try { cameraRef.current?.stopRecording(); } catch {}
    };
  }, [permission, scanFrame]);

  const pickFromGallery = useCallback(async () => {
    clearTimeout(intervalRef.current);
    scanning.current = false;

    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) { resetToScanning(); return; }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 1,
    });

    if (result.canceled) {
      resetToScanning();
      intervalRef.current = setTimeout(scanFrame, 500);
      return;
    }

    try {
      setStatus("processing");
      const data = await sendSingleFrame(result.assets[0].uri);
      showFound(data.url, data.label);
    } catch {
      resetToScanning();
      intervalRef.current = setTimeout(scanFrame, 500);
    }
  }, [sendSingleFrame, showFound, resetToScanning, scanFrame]);

  const handleGo = useCallback(() => {
    if (foundUrl) Linking.openURL(foundUrl);
  }, [foundUrl]);

  const handleBack = useCallback(() => {
    resetToScanning();
    intervalRef.current = setTimeout(scanFrame, 500);
  }, [resetToScanning, scanFrame]);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permBox}>
          <Text style={styles.permTitle}>Kamera İzni Gerekli</Text>
          <Text style={styles.permSub}>
            Ekrandaki görünmez filigranı okumak için kamera erişimi gerekiyor.
          </Text>
          <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
            <Text style={styles.permBtnText}>İzin Ver</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera full screen */}
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="back"
        zoom={0.0}
        autofocus="on"
        mode="video"
      />

      {/* Top — notch area with Brandion */}
      <SafeAreaView edges={["top"]} style={styles.topSafe}>
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.backBtn}
            onPress={() => { clearInterval(intervalRef.current); navigation.goBack(); }}>
            <Text style={styles.backBtnText}>←</Text>
          </TouchableOpacity>
          <View style={styles.notchPill}>
            <Text style={styles.topTitle}>Brand<Text style={styles.topTitleAccent}>ion</Text></Text>
          </View>
          <View style={{ width: 40 }} />
        </View>
      </SafeAreaView>

      {/* Viewfinder center */}
      <View style={styles.viewfinderWrap}>
        <View style={styles.viewfinder}>
          <View style={[styles.corner, styles.tl]} />
          <View style={[styles.corner, styles.tr]} />
          <View style={[styles.corner, styles.bl]} />
          <View style={[styles.corner, styles.br]} />
          {status === "found" && (
            <View style={styles.foundOverlay}>
              <Text style={styles.foundCheck}>✓</Text>
            </View>
          )}
          {(status === "scanning" || status === "processing") && (
            <View style={styles.pillInside}>
              <Animated.View style={[styles.dot, status === "processing" ? styles.dotYellow : undefined, { opacity: dotAnim }]} />
              <Text style={styles.pillText}>
                {status === "scanning" ? "Taranıyor…" : "Analiz ediliyor…"}
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Bottom — home bar area */}
      <SafeAreaView edges={["bottom"]} style={styles.bottomSafe}>
        {status === "found" && (
          <View style={styles.popup}>
            <View style={styles.popupIconRow}>
              <View style={styles.popupIconBadge}>
                <Text style={styles.popupIcon}>✦</Text>
              </View>
            </View>
            <Text style={styles.popupLabel} numberOfLines={2}>{foundLabel}</Text>
            <Text style={styles.popupSub}>Filigran tespit edildi</Text>
            <View style={styles.popupButtons}>
              <TouchableOpacity style={styles.btnBack} onPress={handleBack} activeOpacity={0.7}>
                <Text style={styles.btnBackText}>Geri</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.btnGo} onPress={handleGo} activeOpacity={0.85}>
                <Text style={styles.btnGoText}>Siteye Git  →</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
        {status !== "found" && (
          <View style={styles.bottomBar}>
            <TouchableOpacity style={styles.galleryBtn} onPress={pickFromGallery} activeOpacity={0.75}>
              <Text style={styles.galleryIcon}>⊕</Text>
              <Text style={styles.galleryBtnText}>Galeriden Seç</Text>
            </TouchableOpacity>
            <View style={styles.homeIndicator} />
          </View>
        )}
      </SafeAreaView>
    </View>
  );
}

const VF = 200;
const CORNER = 22;
const CB = 3;
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },

  topSafe: { backgroundColor: "rgba(0,0,0,0.55)" },
  topBar: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    paddingHorizontal: 16, paddingVertical: 10,
  },
  notchPill: {
    backgroundColor: "rgba(0,0,0,0.45)",
    paddingHorizontal: 18, paddingVertical: 6,
    borderRadius: 20, borderWidth: 1, borderColor: "rgba(255,255,255,0.08)",
  },

  bottomSafe: { backgroundColor: "rgba(0,0,0,0.55)" },
  bottomBar: { alignItems: "center", paddingVertical: 12, gap: 10 },
  homeIndicator: {
    width: 120, height: 4, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.3)",
  },
  backBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.08)",
    alignItems: "center", justifyContent: "center",
  },
  backBtnText: { color: "#f0f0f0", fontSize: 18 },
  topTitle: { fontSize: 18, fontWeight: "700", color: "#f0f0f0", letterSpacing: -0.5 },
  topTitleAccent: { color: "#7c6af5" },

  viewfinderWrap: { flex: 1, alignItems: "center", justifyContent: "center" },
  viewfinder: { width: VF, height: VF, borderRadius: 4, overflow: "hidden", position: "relative" },
  corner: { position: "absolute", width: CORNER, height: CORNER, borderColor: "#7c6af5", borderWidth: CB },
  tl: { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0 },
  tr: { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0 },
  bl: { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0 },
  br: { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0 },
  foundOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(74,222,128,0.15)",
    alignItems: "center", justifyContent: "center",
  },
  foundCheck: { fontSize: 72, color: "#4ade80" },

  pillInside: {
    position: "absolute", bottom: 14, alignSelf: "center",
    flexDirection: "row", alignItems: "center", gap: 8,
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 16, paddingVertical: 8,
    borderRadius: 30, borderWidth: 1, borderColor: "rgba(255,255,255,0.1)",
  },
  pill: {
    flexDirection: "row", alignItems: "center", gap: 8,
    backgroundColor: "rgba(255,255,255,0.06)",
    paddingHorizontal: 20, paddingVertical: 11,
    borderRadius: 30, borderWidth: 1, borderColor: "rgba(255,255,255,0.08)",
  },
  pillText: { color: "#f0f0f0", fontSize: 14, fontWeight: "500" },
  dot: {
    width: 7, height: 7, borderRadius: 4, backgroundColor: "#7c6af5",
    shadowColor: "#7c6af5", shadowOpacity: 1, shadowRadius: 4,
    shadowOffset: { width: 0, height: 0 },
  },
  dotYellow: { backgroundColor: "#facc15", shadowColor: "#facc15" },

  popup: {
    backgroundColor: "rgba(12,10,24,0.97)",
    borderRadius: 28, paddingHorizontal: 24, paddingVertical: 28,
    width: "100%", gap: 12,
    borderWidth: 1, borderColor: "rgba(124,106,245,0.25)",
    shadowColor: "#7c6af5", shadowOpacity: 0.3, shadowRadius: 24,
    shadowOffset: { width: 0, height: 0 },
  },
  popupIconRow: { alignItems: "center" },
  popupIconBadge: {
    width: 44, height: 44, borderRadius: 14,
    backgroundColor: "rgba(124,106,245,0.15)",
    borderWidth: 1, borderColor: "rgba(124,106,245,0.3)",
    alignItems: "center", justifyContent: "center",
  },
  popupIcon: { fontSize: 20, color: "#7c6af5" },
  popupLabel: { color: "#f0f0f0", fontSize: 16, fontWeight: "700", textAlign: "center", lineHeight: 22 },
  popupSub: { color: "#4ade80", fontSize: 12, fontWeight: "500", textAlign: "center", letterSpacing: 0.5 },
  popupButtons: { flexDirection: "row", gap: 10, marginTop: 4 },
  btnBack: {
    flex: 1, paddingVertical: 15, borderRadius: 16,
    backgroundColor: "rgba(255,255,255,0.06)",
    borderWidth: 1, borderColor: "rgba(255,255,255,0.08)",
    alignItems: "center",
  },
  btnBackText: { color: "#666", fontWeight: "600", fontSize: 14 },
  btnGo: {
    flex: 2, paddingVertical: 15, borderRadius: 16,
    backgroundColor: "#7c6af5", alignItems: "center",
    shadowColor: "#7c6af5", shadowOpacity: 0.5, shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
  },
  btnGoText: { color: "#fff", fontWeight: "700", fontSize: 15, letterSpacing: 0.3 },

  galleryBtn: {
    flexDirection: "row", alignItems: "center", gap: 8,
    paddingVertical: 13, paddingHorizontal: 28, borderRadius: 30,
    borderWidth: 1, borderColor: "rgba(124,106,245,0.35)",
    backgroundColor: "rgba(124,106,245,0.08)",
  },
  galleryIcon: { color: "#7c6af5", fontSize: 16 },
  galleryBtnText: { color: "#a89ef5", fontSize: 14, fontWeight: "600", letterSpacing: 0.2 },
  hint: { color: "#444", fontSize: 11, letterSpacing: 0.3 },

  permBox: { flex: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: 40, gap: 16 },
  permTitle: { fontSize: 22, fontWeight: "700", color: "#f0f0f0" },
  permSub: { fontSize: 15, color: "#666", textAlign: "center", lineHeight: 22 },
  permBtn: {
    backgroundColor: "#7c6af5", borderRadius: 12,
    paddingHorizontal: 32, paddingVertical: 14, marginTop: 8,
  },
  permBtnText: { color: "#fff", fontWeight: "700", fontSize: 16 },
});
