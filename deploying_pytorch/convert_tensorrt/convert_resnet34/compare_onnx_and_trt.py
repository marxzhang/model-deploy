# 在 Environment variables 里点击文件夹图标，添加：
# - Name: LD_LIBRARY_PATH
# - Value: /usr/local/TensorRT-8.6/lib


import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRUNING_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", "pruning_model_pytorch"))
if PRUNING_DIR not in sys.path:
    sys.path.insert(0, PRUNING_DIR)
# from model import resnet34 as create_resnet34
from torchvision.models.resnet import resnet34 as create_model
from utils import evaluate

WEIGHTS_PTH = os.path.join(PRUNING_DIR, "resNet34.pth")
PRUNING_PTH = os.path.join(PRUNING_DIR, "pruning_model.pth")
ONNX_PATH = os.path.join(BASE_DIR, "resnet34.onnx")
TRT_PATH = os.path.join(BASE_DIR, "trt_output", "resnet34.trt")
VAL_ROOT = os.path.expanduser("~/code/data/flower_data/val")

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

DATA_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


def build_val_loader(batch_size=8):
    val_dataset = datasets.ImageFolder(root=VAL_ROOT, transform=DATA_TRANSFORM)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader


def load_pytorch_model(weights_path: str) -> nn.Module:
    model = create_model(num_classes=5)
    state_dict = torch.load(weights_path, map_location="cpu")
    if any(k.endswith(".weight_orig") for k in state_dict):
        # 权重由 torch.nn.utils.prune 保存: conv weight = weight_orig * weight_mask
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_orig = state_dict.pop(name + ".weight_orig", None)
                weight_mask = state_dict.pop(name + ".weight_mask", None)
                if weight_orig is not None and weight_mask is not None:
                    module.weight = nn.Parameter(weight_orig * weight_mask.to(weight_orig.dtype))
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


class OnnxWrapper(nn.Module):
    """Wrap ONNX Runtime session as a torch module so utils.evaluate can be reused."""

    def __init__(self, ort_session):
        super().__init__()
        self.ort_session = ort_session
        self.input_name = ort_session.get_inputs()[0].name

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        outs = [self.ort_session.run(None, {self.input_name: x_np[i:i + 1]})[0] for i in range(x_np.shape[0])]
        return torch.from_numpy(np.concatenate(outs, axis=0)).to(x.device)


class TrtWrapper(nn.Module):
    """Wrap a TensorRT engine as a torch module so utils.evaluate can be reused."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.stream = cuda.Stream()

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        batch_outs = []
        with self.engine.create_execution_context() as context:
            for i in range(x_np.shape[0]):
                image = np.ascontiguousarray(x_np[i:i + 1])
                context.set_binding_shape(self.engine.get_binding_index("input"), image.shape)
                bindings = []
                for binding in self.engine:
                    binding_idx = self.engine.get_binding_index(binding)
                    size = trt.volume(context.get_binding_shape(binding_idx))
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    if self.engine.binding_is_input(binding):
                        input_buffer = image
                        input_memory = cuda.mem_alloc(image.nbytes)
                        bindings.append(int(input_memory))
                    else:
                        output_buffer = cuda.pagelocked_empty(size, dtype)
                        output_memory = cuda.mem_alloc(output_buffer.nbytes)
                        bindings.append(int(output_memory))

                cuda.memcpy_htod_async(input_memory, input_buffer, self.stream)
                context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                cuda.memcpy_dtoh_async(output_buffer, output_memory, self.stream)
                self.stream.synchronize()
                batch_outs.append(np.reshape(output_buffer, (1, -1)))

        return torch.from_numpy(np.concatenate(batch_outs, axis=0)).to(x.device)


def main():
    print("== loading validation set from {} ==".format(VAL_ROOT))
    val_loader = build_val_loader(batch_size=8)

    print("== loading models ==")
    models = {
        "resNet34.pth (原始)": load_pytorch_model(WEIGHTS_PTH),
        "pruning_model.pth (剪枝)": load_pytorch_model(PRUNING_PTH),
        "resnet34.onnx": OnnxWrapper(onnxruntime.InferenceSession(ONNX_PATH)),
    }

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(TRT_PATH, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    models["resnet34.trt"] = TrtWrapper(engine)

    for name, model in models.items():
        if isinstance(model, (OnnxWrapper, TrtWrapper)):
            model.eval()
        else:
            model.cuda().eval()

    print("== evaluating on validation set ==")
    results = {}
    for name, model in models.items():
        loss, acc = evaluate(model=model, data_loader=val_loader, epoch=0)
        results[name] = (loss, acc)

    print("\n========== 验证集精度对比 (Top-1 Acc) ==========")
    for name, (loss, acc) in results.items():
        print("{:<30} loss={:.4f}   acc={:.4f}".format(name, loss, acc))


if __name__ == '__main__':
    main()
