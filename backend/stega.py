import os
from typing import Optional

import bchlib
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image, ImageOps

tf.disable_v2_behavior()

from tensorflow.python.saved_model import tag_constants, signature_constants

try:
    import tensorflow_addons  # TF1 contrib op'larini kaydet (varsa)
except ImportError:
    pass

BCH_POLYNOMIAL = 137
BCH_BITS = 5
SECRET_SIZE = 100
IMAGE_SIZE = 400
MODEL_REPO = "KingTechnician/stegastamp"
MODEL_CACHE = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")


class StegaStamp:
    def __init__(self):
        self.sess = None
        self.input_secret = None
        self.input_image = None
        self.output_stegastamp = None
        self.output_decoded = None
        self._bch = None
        self._load()

    def _get_model_path(self) -> str:
        model_path = os.path.join(MODEL_CACHE, "stegastamp_pretrained")
        if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
            print("Model bulunamadi, HuggingFace'den indiriliyor (~217MB)...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=MODEL_CACHE,
                ignore_patterns=["*.pth", "encoder.py", "decoder.py",
                                  "stegastamp_model.py", "example_usage.py",
                                  "requirements.txt", "config.json"],
            )
            print("Model indirildi.")
        return model_path

    def _load(self):
        model_path = self._get_model_path()
        self.sess = tf.Session(graph=tf.Graph())
        with self.sess.graph.as_default():
            model = tf.saved_model.loader.load(
                self.sess, [tag_constants.SERVING], model_path
            )
            sig = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            print("[STEGA] inputs:", list(sig.inputs.keys()))
            print("[STEGA] outputs:", list(sig.outputs.keys()))
            self.input_secret = self.sess.graph.get_tensor_by_name(
                sig.inputs["secret"].name
            )
            self.input_image = self.sess.graph.get_tensor_by_name(
                sig.inputs["image"].name
            )
            self.output_stegastamp = self.sess.graph.get_tensor_by_name(
                sig.outputs["stegastamp"].name
            )
            self.output_residual = self.sess.graph.get_tensor_by_name(
                sig.outputs["residual"].name
            )
            self.output_decoded = self.sess.graph.get_tensor_by_name(
                sig.outputs["decoded"].name
            )
        print("StegaStamp modeli basariyla yuklendi.")

    @property
    def bch(self):
        if self._bch is None:
            for args, kwargs in [
                ([], {"prim_poly": BCH_POLYNOMIAL, "t": BCH_BITS}),
                ([], {"polynomial": BCH_POLYNOMIAL, "t": BCH_BITS}),
                ([BCH_POLYNOMIAL, BCH_BITS], {}),
            ]:
                try:
                    self._bch = bchlib.BCH(*args, **kwargs)
                    break
                except TypeError:
                    continue
        return self._bch

    def _prepare_01(self, image: Image.Image) -> np.ndarray:
        """[0,1] normalizasyon"""
        image = image.convert("RGB")
        image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE))
        return np.array(image, dtype=np.float32) / 255.0

    def _prepare(self, image: Image.Image) -> np.ndarray:
        return self._prepare_01(image)

    def encode(self, image: Image.Image, code: str) -> Image.Image:
        code = (code[:7] + " " * 7)[:7]
        data = bytearray(code, "utf-8")
        ecc = self.bch.encode(data)
        packet = data + ecc
        bits = [int(b) for byte in packet for b in format(byte, "08b")]
        bits += [0] * (SECRET_SIZE - len(bits))

        img_np = self._prepare_01(image)

        stegastamp, residual = self.sess.run(
            [self.output_stegastamp, self.output_residual],
            feed_dict={self.input_secret: [bits], self.input_image: [img_np]},
        )
        res = residual[0]
        print(f"[STEGA] residual min={res.min():.4f} max={res.max():.4f} mean={res.mean():.4f} std={res.std():.4f}")

        # Orijinal gorsel + perturbation (x2 = kamera icin daha guclü ama hala gorunmez)
        out = np.clip(img_np + 2.0 * res, 0, 1)
        diff = np.abs(res).mean()
        print(f"[STEGA] pixel diff mean={diff:.4f}")
        return Image.fromarray((out * 255).astype(np.uint8))

    def decode(self, image: Image.Image) -> Optional[str]:
        img_np = self._prepare(image)
        raw = self.sess.run(
            self.output_decoded,
            feed_dict={self.input_image: [img_np]},
        )[0]

        packet_binary = "".join(str(int(b > 0.5)) for b in raw[:96])
        packet = bytearray(
            int(packet_binary[i: i + 8], 2) for i in range(0, 96, 8)
        )
        data, ecc = packet[: -self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]

        try:
            bitflips = self.bch.decode_inplace(data, ecc)
        except AttributeError:
            bitflips = self.bch.decode(data, ecc)

        if bitflips < 0:
            return None

        try:
            return data.decode("utf-8").strip()
        except Exception:
            return None
