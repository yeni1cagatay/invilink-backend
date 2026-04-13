import { useNavigation } from "@react-navigation/native";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

export default function HomeScreen() {
  const navigation = useNavigation();

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.inner}>
        <View style={styles.logoWrap}>
          <Text style={styles.logoText}>
            Brand<Text style={styles.logoAccent}>ion</Text>
          </Text>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>BETA</Text>
          </View>
        </View>

        <Text style={styles.headline}>Görünmez{"\n"}Alışveriş</Text>
        <Text style={styles.sub}>
          Kameranı ekrana tut.{"\n"}Gizli bağlantı otomatik bulunur.
        </Text>

        <TouchableOpacity
          style={styles.scanBtn}
          onPress={() => navigation.navigate("Scan")}
          activeOpacity={0.85}
        >
          <Text style={styles.scanIcon}>◎</Text>
          <Text style={styles.scanBtnText}>Taramayı Başlat</Text>
        </TouchableOpacity>

        <View style={styles.steps}>
          {[
            { n: "1", t: "Uygulamayı aç" },
            { n: "2", t: "Kamerayı ekrana tut" },
            { n: "3", t: "Ürün sayfası açılır" },
          ].map((s) => (
            <View key={s.n} style={styles.step}>
              <View style={styles.stepNum}>
                <Text style={styles.stepNumText}>{s.n}</Text>
              </View>
              <Text style={styles.stepText}>{s.t}</Text>
            </View>
          ))}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  inner: { flex: 1, paddingHorizontal: 28, paddingTop: 24 },
  logoWrap: { flexDirection: "row", alignItems: "center", gap: 10, marginBottom: 48 },
  logoText: { fontSize: 22, fontWeight: "700", color: "#f0f0f0", letterSpacing: -0.5 },
  logoAccent: { color: "#7c6af5" },
  badge: {
    backgroundColor: "#1a1a2e", paddingHorizontal: 10, paddingVertical: 3,
    borderRadius: 20, borderWidth: 1, borderColor: "#3a3060",
  },
  badgeText: { color: "#7c6af5", fontSize: 10, fontWeight: "700" },
  headline: {
    fontSize: 42, fontWeight: "800", color: "#f0f0f0",
    letterSpacing: -1, lineHeight: 50, marginBottom: 16,
  },
  sub: { fontSize: 16, color: "#666", lineHeight: 24, marginBottom: 48 },
  scanBtn: {
    backgroundColor: "#7c6af5", borderRadius: 16,
    paddingVertical: 20, paddingHorizontal: 28,
    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 12,
    marginBottom: 48,
  },
  scanIcon: { fontSize: 22, color: "#fff" },
  scanBtnText: { fontSize: 18, fontWeight: "700", color: "#fff" },
  steps: { gap: 16 },
  step: { flexDirection: "row", alignItems: "center", gap: 14 },
  stepNum: {
    width: 32, height: 32, borderRadius: 16,
    backgroundColor: "#1a1a2e", borderWidth: 1, borderColor: "#3a3060",
    alignItems: "center", justifyContent: "center",
  },
  stepNumText: { color: "#7c6af5", fontWeight: "700", fontSize: 14 },
  stepText: { color: "#888", fontSize: 15 },
});
