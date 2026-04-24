import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { useEffect, useRef } from "react";
import { Animated, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

// B r a n → mor | d i → geçiş | o n → beyaz
const LOGO_LETTERS = [
  { char: "B", color: "#7c6af5" },
  { char: "r", color: "#7c6af5" },
  { char: "a", color: "#7c6af5" },
  { char: "n", color: "#7c6af5" },
  { char: "d", color: "#9d8df8" },
  { char: "i", color: "#e8e4fd" },
  { char: "o", color: "#f0f0f0" },
  { char: "n", color: "#f0f0f0" },
];

export default function HomeScreen() {
  const navigation = useNavigation();
  const glowAnim = useRef(new Animated.Value(0.5)).current;
  const dotAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(24)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, { toValue: 1, duration: 2400, useNativeDriver: true }),
        Animated.timing(glowAnim, { toValue: 0.5, duration: 2400, useNativeDriver: true }),
      ])
    ).start();

    // yeşil nokta nabız
    Animated.loop(
      Animated.sequence([
        Animated.timing(dotAnim, { toValue: 0.2, duration: 800, useNativeDriver: true }),
        Animated.timing(dotAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();

    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 700, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 700, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <Animated.View
        style={[styles.inner, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}
      >

        {/* ── ÜST: Brandion yazısı ── */}
        <View style={styles.topLogoBlock}>
          <View style={styles.logoRow}>
            {LOGO_LETTERS.map((l, i) => (
              <Text key={i} style={[styles.logoChar, { color: l.color }]}>
                {l.char}
              </Text>
            ))}
          </View>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>BETA</Text>
          </View>
        </View>

        {/* ── ORTA: Parlayan halka ── */}
        <View style={styles.circleBlock}>
          <Animated.View style={[styles.glowRingOuter, { opacity: glowAnim }]} />
          <View style={styles.glowRingInner}>
            <Ionicons name="camera" size={48} color="#7c6af5" />
            {/* yeşil aktif nokta */}
            <Animated.View style={[styles.greenDot, { opacity: dotAnim }]} />
          </View>
        </View>

        {/* ── TAGLINE ── */}
        <View style={styles.taglineBlock}>
          <View style={styles.taglineRow}>
            <Animated.View style={[styles.liveDot, { opacity: dotAnim }]} />
            <Text style={styles.tagline}>Ekrana Tut</Text>
          </View>
          <Text style={styles.sub}>Kameranı ekrana tut.</Text>
        </View>

        <View style={{ flex: 1, minHeight: 24 }} />

        {/* ── CTA ── */}
        <TouchableOpacity
          style={styles.scanBtn}
          onPress={() => navigation.navigate("Scan")}
          activeOpacity={0.85}
        >
          <Text style={styles.scanBtnText}>Taramayı Başlat</Text>
          <Text style={styles.scanArrow}>→</Text>
        </TouchableOpacity>

        {/* ── 3 ADIM ── */}
        <View style={styles.steps}>
          {[
            { n: "1", t: "Uygulamayı aç", green: false },
            { n: "2", t: "Kamerayı ekrana tut", green: false },
            { n: "3", t: "Ürün sayfası açılır", green: true },
          ].map((s) => (
            <View key={s.n} style={styles.step}>
              <View style={[styles.stepNum, s.green && styles.stepNumGreen]}>
                <Text style={[styles.stepNumText, s.green && styles.stepNumTextGreen]}>
                  {s.n}
                </Text>
              </View>
              <Text style={[styles.stepText, s.green && styles.stepTextGreen]}>{s.t}</Text>
            </View>
          ))}
        </View>

      </Animated.View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  inner: {
    flex: 1,
    paddingHorizontal: 28,
    paddingTop: 32,
    paddingBottom: 24,
    alignItems: "center",
  },

  // ── ÜST LOGO ──
  topLogoBlock: {
    alignItems: "center",
    gap: 10,
    marginBottom: 52,
  },
  logoRow: {
    flexDirection: "row",
    alignItems: "baseline",
  },
  logoChar: {
    fontSize: 48,
    fontWeight: "800",
    fontFamily: "Arial",
    letterSpacing: 0.5,
  },
  badge: {
    backgroundColor: "#1a1a2e",
    paddingHorizontal: 14,
    paddingVertical: 4,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "#3a3060",
  },
  badgeText: {
    color: "#7c6af5",
    fontSize: 10,
    fontWeight: "700",
    fontFamily: "Arial",
    letterSpacing: 2,
  },

  // ── ORTA KARE ──
  circleBlock: {
    alignItems: "center",
    justifyContent: "center",
    width: 200,
    height: 200,
    marginBottom: 36,
  },
  glowRingOuter: {
    position: "absolute",
    width: 200,
    height: 200,
    borderRadius: 32,
    borderWidth: 1,
    borderColor: "#7c6af5",
    shadowColor: "#7c6af5",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 40,
  },
  glowRingInner: {
    width: 120,
    height: 120,
    borderRadius: 20,
    backgroundColor: "#13102a",
    borderWidth: 1,
    borderColor: "#3a3060",
    alignItems: "center",
    justifyContent: "center",
  },
  greenDot: {
    position: "absolute",
    bottom: 14,
    right: 14,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#4ade80",
    shadowColor: "#4ade80",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 6,
  },

  // ── TAGLINE ──
  taglineBlock: {
    alignItems: "center",
    gap: 10,
  },
  taglineRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#4ade80",
    shadowColor: "#4ade80",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 6,
  },
  tagline: {
    fontSize: 22,
    fontWeight: "700",
    fontFamily: "Arial",
    color: "#f0f0f0",
    letterSpacing: -0.3,
  },
  sub: {
    fontSize: 15,
    fontFamily: "Arial",
    color: "#aaa",
    lineHeight: 23,
    textAlign: "center",
  },

  // ── CTA ──
  scanBtn: {
    backgroundColor: "#7c6af5",
    borderRadius: 16,
    paddingVertical: 18,
    paddingHorizontal: 28,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    marginBottom: 28,
    width: "100%",
  },
  scanBtnText: {
    fontSize: 17,
    fontWeight: "700",
    fontFamily: "Arial",
    color: "#fff",
  },
  scanArrow: {
    fontSize: 18,
    color: "rgba(255,255,255,0.6)",
  },

  // ── ADIMLAR ──
  steps: {
    flexDirection: "row",
    justifyContent: "space-between",
    width: "100%",
  },
  step: { alignItems: "center", gap: 8, flex: 1 },
  stepNum: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#1a1a2e",
    borderWidth: 1,
    borderColor: "#3a3060",
    alignItems: "center",
    justifyContent: "center",
  },
  stepNumText: {
    color: "#7c6af5",
    fontWeight: "700",
    fontFamily: "Arial",
    fontSize: 13,
  },
  stepText: {
    color: "#aaa",
    fontSize: 12,
    fontFamily: "Arial",
    textAlign: "center",
    lineHeight: 16,
  },
  stepNumGreen: {
    backgroundColor: "#0d2b1a",
    borderColor: "#1a5c30",
  },
  stepNumTextGreen: {
    color: "#4ade80",
  },
  stepTextGreen: {
    color: "#4ade80",
    opacity: 0.8,
  },
});
