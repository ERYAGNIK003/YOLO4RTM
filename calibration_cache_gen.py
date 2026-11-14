import os
import cv2
import yaml
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# =========================
# CONFIG
# =========================
ONNX_PATH = "/home/seas/YOLOv5OBB_github/yolov5_obb/runs/train/yolo_e100_scratch_hyp_lessaug_n_ablation_kld/weights/best.onnx"
CACHE_FILE = "calibration.cache"
DATA_YAML = "/home/seas/YOLOv5OBB_github/yolov5_obb/dataset/DRASHTI-HaOBB-yolo5.yaml"   # your dataset yaml
IMG_SIZE = 1024
STRIDE = 32
BATCH_SIZE = 1


# =========================
# LETTERBOX (YOLOv5 style)
# =========================
def letterbox(img, new_shape=(1024, 1024), stride=32, color=(114, 114, 114)):
    shape = img.shape[:2]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    return img


# =========================
# LOAD IMAGES FROM YAML
# =========================
def load_image_paths(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    paths = []

    for split in ['train', 'val']:  # use both
        if split in data:
            p = data[split]

            if isinstance(p, list):
                for sub in p:
                    paths += get_images(sub)
            else:
                paths += get_images(p)

    return paths


def get_images(path):
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                imgs.append(os.path.join(root, f))
    return imgs


# =========================
# CALIBRATOR
# =========================
class YOLOv5Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths):
        super().__init__()

        self.image_paths = image_paths
        self.batch_size = BATCH_SIZE
        self.current_index = 0

        self.input_shape = (3, IMG_SIZE, IMG_SIZE)

        self.device_input = cuda.mem_alloc(
            BATCH_SIZE * np.prod(self.input_shape) * np.float32().nbytes
        )

    def get_batch_size(self):
        return self.batch_size

    def preprocess(self, img_path):
        img0 = cv2.imread(img_path)

        img = letterbox(img0, (IMG_SIZE, IMG_SIZE), stride=STRIDE)

        img = img.transpose((2, 0, 1))[::-1]  # BGR → RGB
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32) / 255.0  # 🔥 critical

        return img

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)

        for i in range(self.batch_size):
            if self.current_index >= len(self.image_paths):
                break

            img = self.preprocess(self.image_paths[self.current_index])
            batch[i] = img
            self.current_index += 1

        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(CACHE_FILE, "wb") as f:
            f.write(cache)
        print(f"✅ Calibration cache saved: {CACHE_FILE}")


# =========================
# BUILD ENGINE (to trigger calibration)
# =========================
def build_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30

        config.set_flag(trt.BuilderFlag.INT8)

        image_paths = load_image_paths(DATA_YAML)

        # optional: limit images
        image_paths = image_paths[:1000]

        print(f"Using {len(image_paths)} images for calibration")

        calibrator = YOLOv5Calibrator(image_paths)
        config.int8_calibrator = calibrator

        print("⏳ Building engine + generating calibration cache...")

        builder.build_engine(network, config)

        print("✅ Done")


if __name__ == "__main__":
    build_engine()
