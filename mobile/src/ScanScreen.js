import { useNavigation } from "@react-navigation/native";
import axios from "axios";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import * as Linking from "expo-linking";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  StyleSheet,
  Text,
  TouchableOpacity,
  Vibration,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { API_URL } from "./config";

const SCAN_INTERVAL_MS = 1500; // Her 1.5 saniyede bir frame gonder

export default function ScanScreen() {
  const navigation = useNavigation();
  const [permission, requestPermission] = useCameraPermissions();
  const [status, setStatus] = useState("scanning"); // scanning | found | error | no_watermark
  const [foundUrl, setFoundUrl] = useState(null);
  const cameraRef = useRef(null);
  const scanning = useRef(false);
  const intervalRef = useRef(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Tarama animasyonu
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.08, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  const sendFrame = useCallback(async () => {
    if (scanning.current || !cameraRef.current) return;
    scanning.current = true;

    try {
      // Frame yakala
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        skipProcessing: true,
      });

      // 400x400'e yeniden boyutlandir (model bu boyutu bekliyor)
      const resized = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 400, height: 400 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );

      // Sunucuya gonder
      const form = new FormData();
      form.append("image", {
        uri: resized.uri,
        type: "image/jpeg",
        name: "frame.jpg",
      });

      const res = await axios.post(`${API_URL}/api/decode`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 8000,
      });

      // Filigran bulundu!
      Vibration.vibrate(200);
      clearInterval(intervalRef.current);
      setFoundUrl(res.data.url);
      setStatus("found");

      // 1 saniye sonra tarayiciyi ac
      setTimeout(() => {
        Linking.openURL(res.data.url);
      }, 1000);
    } catch (err) {
      if (err.response?.status === 404) {
        // Filigran yok — taramaya devam
        setStatus("scanning");
      } else {
        // Baglanti hatasi vb.
        setStatus("error");
        clearInterval(intervalRef.current);
      }
    } finally {
      scanning.current = false;
    }
  }, []);

  // Kamera izni alininca taramayı baslat
  useEffect(() => {
    if (!permission?.granted) return;
    intervalRef.current = setInterval(sendFrame, SCAN_INTERVAL_MS);
    return () => clearInterval(intervalRef.current);
  }, [permission, sendFrame]);

  // --- Izin ekrani ---
  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permBox}>
          <Text style={styles.permTitle}>Kamera İzni Gerekli</Text>
          <Text style={styles.permSub}>
            Ekrandaki görünmez filigranı okumak için kamera erişimine ihtiyacımız var.
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
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />

      {/* Karartma + Kılavuz Çerçeve */}
      <View style={styles.overlay}>
        <View style={styles.topMask} />
        <View style={styles.middleRow}>
          <View style={styles.sideMask} />
          <Animated.View
            style={[styles.viewfinder, { transform: [{ scale: pulseAnim }] }]}
          >
            <View style={[styles.corner, styles.tl]} />
            <View style={[styles.corner, styles.tr]} />
            <View style={[styles.corner, styles.bl]} />
            <View style={[styles.corner, styles.br]} />
          </Animated.View>
          <View style={styles.sideMask} />
        </View>
        <View style={styles.bottomMask}>
          <StatusBar status={status} foundUrl={foundUrl} />
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => { clearInterval(intervalRef.current); navigation.goBack(); }}
          >
            <Text style={styles.backBtnText}>← Geri</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

function StatusBar({ status, foundUrl }) {
  if (status === "found") {
    return (
      <View style={[styles.statusBox, styles.statusFound]}>
        <Text style={styles.statusIcon}>✓</Text>
        <Text style={styles.statusText}>Filigran bulundu! Açılıyor...</Text>
      </View>
    );
  }
  if (status === "error") {
    return (
      <View style={[styles.statusBox, styles.statusError]}>
        <Text style={styles.statusIcon}>⚠</Text>
        <Text style={styles.statusText}>Bağlantı hatası. Tekrar dene.</Text>
      </View>
    );
  }
  return (
    <View style={styles.statusBox}>
      <ActivityIndicator color="#7c6af5" size="small" />
      <Text style={styles.statusText}>Ekrana tut, taranıyor...</Text>
    </View>
  );
}

const CORNER = 24;
const BORDER = 3;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  overlay: { flex: 1 },
  topMask: { flex: 1, backgroundColor: "rgba(0,0,0,0.6)" },
  middleRow: { flexDirection: "row", height: 260 },
  sideMask: { flex: 1, backgroundColor: "rgba(0,0,0,0.6)" },
  bottomMask: {
    flex: 1, backgroundColor: "rgba(0,0,0,0.6)",
    alignItems: "center", justifyContent: "flex-start", paddingTop: 24, gap: 16,
  },
  viewfinder: {
    width: 260, height: 260,
    borderWidth: 0,
  },
  corner: {
    position: "absolute", width: CORNER, height: CORNER,
    borderColor: "#7c6af5", borderWidth: BORDER,
  },
  tl: { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0 },
  tr: { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0 },
  bl: { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0 },
  br: { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0 },
  statusBox: {
    flexDirection: "row", alignItems: "center", gap: 10,
    backgroundColor: "rgba(255,255,255,0.08)",
    paddingHorizontal: 20, paddingVertical: 12,
    borderRadius: 30,
  },
  statusFound: { backgroundColor: "rgba(76,175,80,0.2)" },
  statusError: { backgroundColor: "rgba(244,67,54,0.2)" },
  statusIcon: { fontSize: 16 },
  statusText: { color: "#f0f0f0", fontSize: 14, fontWeight: "500" },
  backBtn: { paddingHorizontal: 20, paddingVertical: 10 },
  backBtnText: { color: "#888", fontSize: 15 },
  permBox: {
    flex: 1, alignItems: "center", justifyContent: "center",
    paddingHorizontal: 40, gap: 16,
  },
  permTitle: { fontSize: 22, fontWeight: "700", color: "#f0f0f0" },
  permSub: { fontSize: 15, color: "#888", textAlign: "center", lineHeight: 22 },
  permBtn: {
    backgroundColor: "#7c6af5", borderRadius: 12,
    paddingHorizontal: 32, paddingVertical: 14, marginTop: 8,
  },
  permBtnText: { color: "#fff", fontWeight: "700", fontSize: 16 },
});
