# pnnx model stat
# model inputshape = [1,192,32]f32,[1]i64,[1]f32
# FLOPS = 349.115M
# memory OPS = 4.512M

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mel_adapter_0 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=192, kernel_size=(5,), out_channels=256, padding=(2,), padding_mode='zeros', stride=(1,))
        self.mel_adapter_1 = nn.ReLU()
        self.mel_adapter_2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=256, kernel_size=(5,), out_channels=256, padding=(2,), padding_mode='zeros', stride=(1,))
        self.mel_adapter_3 = nn.ReLU()
        self.convnext_blocks_0_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=256, in_channels=256, kernel_size=(7,), out_channels=256, padding=(3,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_0_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=256, num_groups=1)
        self.convnext_blocks_0_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=256, kernel_size=(1,), out_channels=1024, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_0_act = nn.GELU()
        self.convnext_blocks_0_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=1024, kernel_size=(1,), out_channels=256, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_1_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=256, in_channels=256, kernel_size=(7,), out_channels=256, padding=(3,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_1_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=256, num_groups=1)
        self.convnext_blocks_1_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=256, kernel_size=(1,), out_channels=1024, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_1_act = nn.GELU()
        self.convnext_blocks_1_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=1024, kernel_size=(1,), out_channels=256, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_2_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=256, in_channels=256, kernel_size=(7,), out_channels=256, padding=(3,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_2_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=256, num_groups=1)
        self.convnext_blocks_2_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=256, kernel_size=(1,), out_channels=1024, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_2_act = nn.GELU()
        self.convnext_blocks_2_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=1024, kernel_size=(1,), out_channels=256, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_3_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=256, in_channels=256, kernel_size=(7,), out_channels=256, padding=(3,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_3_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=256, num_groups=1)
        self.convnext_blocks_3_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=256, kernel_size=(1,), out_channels=1024, padding=(0,), padding_mode='zeros', stride=(1,))
        self.convnext_blocks_3_act = nn.GELU()
        self.convnext_blocks_3_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=1024, kernel_size=(1,), out_channels=256, padding=(0,), padding_mode='zeros', stride=(1,))
        self.mel_head = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=256, kernel_size=(1,), out_channels=80, padding=(0,), padding_mode='zeros', stride=(1,))

        archive = zipfile.ZipFile('D:/project/Android_Projects/TTS/models/duration_prosody_predictor.pnnx.bin', 'r')
        self.mel_adapter_0.bias = self.load_pnnx_bin_as_parameter(archive, 'mel_adapter.0.bias', (256), 'float32')
        self.mel_adapter_0.weight = self.load_pnnx_bin_as_parameter(archive, 'mel_adapter.0.weight', (256,192,5), 'float32')
        self.mel_adapter_2.bias = self.load_pnnx_bin_as_parameter(archive, 'mel_adapter.2.bias', (256), 'float32')
        self.mel_adapter_2.weight = self.load_pnnx_bin_as_parameter(archive, 'mel_adapter.2.weight', (256,256,5), 'float32')
        self.convnext_blocks_0_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.dwconv.bias', (256), 'float32')
        self.convnext_blocks_0_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.dwconv.weight', (256,1,7), 'float32')
        self.convnext_blocks_0_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.norm.bias', (256), 'float32')
        self.convnext_blocks_0_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.norm.weight', (256), 'float32')
        self.convnext_blocks_0_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.pwconv1.bias', (1024), 'float32')
        self.convnext_blocks_0_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.pwconv1.weight', (1024,256,1), 'float32')
        self.convnext_blocks_0_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.pwconv2.bias', (256), 'float32')
        self.convnext_blocks_0_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.0.pwconv2.weight', (256,1024,1), 'float32')
        self.convnext_blocks_1_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.dwconv.bias', (256), 'float32')
        self.convnext_blocks_1_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.dwconv.weight', (256,1,7), 'float32')
        self.convnext_blocks_1_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.norm.bias', (256), 'float32')
        self.convnext_blocks_1_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.norm.weight', (256), 'float32')
        self.convnext_blocks_1_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.pwconv1.bias', (1024), 'float32')
        self.convnext_blocks_1_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.pwconv1.weight', (1024,256,1), 'float32')
        self.convnext_blocks_1_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.pwconv2.bias', (256), 'float32')
        self.convnext_blocks_1_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.1.pwconv2.weight', (256,1024,1), 'float32')
        self.convnext_blocks_2_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.dwconv.bias', (256), 'float32')
        self.convnext_blocks_2_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.dwconv.weight', (256,1,7), 'float32')
        self.convnext_blocks_2_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.norm.bias', (256), 'float32')
        self.convnext_blocks_2_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.norm.weight', (256), 'float32')
        self.convnext_blocks_2_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.pwconv1.bias', (1024), 'float32')
        self.convnext_blocks_2_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.pwconv1.weight', (1024,256,1), 'float32')
        self.convnext_blocks_2_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.pwconv2.bias', (256), 'float32')
        self.convnext_blocks_2_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.2.pwconv2.weight', (256,1024,1), 'float32')
        self.convnext_blocks_3_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.dwconv.bias', (256), 'float32')
        self.convnext_blocks_3_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.dwconv.weight', (256,1,7), 'float32')
        self.convnext_blocks_3_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.norm.bias', (256), 'float32')
        self.convnext_blocks_3_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.norm.weight', (256), 'float32')
        self.convnext_blocks_3_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.pwconv1.bias', (1024), 'float32')
        self.convnext_blocks_3_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.pwconv1.weight', (1024,256,1), 'float32')
        self.convnext_blocks_3_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.pwconv2.bias', (256), 'float32')
        self.convnext_blocks_3_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'convnext_blocks.3.pwconv2.weight', (256,1024,1), 'float32')
        self.mel_head.bias = self.load_pnnx_bin_as_parameter(archive, 'mel_head.bias', (80), 'float32')
        self.mel_head.weight = self.load_pnnx_bin_as_parameter(archive, 'mel_head.weight', (80,256,1), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0, v_1, v_2):
        v_3 = F.upsample(v_0, mode='nearest', scale_factor=(2.0,))
        v_4 = self.mel_adapter_0(v_3)
        v_5 = self.mel_adapter_1(v_4)
        v_6 = self.mel_adapter_2(v_5)
        v_7 = self.mel_adapter_3(v_6)
        v_8 = self.convnext_blocks_0_dwconv(v_7)
        v_9 = self.convnext_blocks_0_norm(v_8)
        v_10 = self.convnext_blocks_0_pwconv1(v_9)
        v_11 = self.convnext_blocks_0_act(v_10)
        v_12 = self.convnext_blocks_0_pwconv2(v_11)
        v_13 = (v_7 + v_12)
        v_14 = self.convnext_blocks_1_dwconv(v_13)
        v_15 = self.convnext_blocks_1_norm(v_14)
        v_16 = self.convnext_blocks_1_pwconv1(v_15)
        v_17 = self.convnext_blocks_1_act(v_16)
        v_18 = self.convnext_blocks_1_pwconv2(v_17)
        v_19 = (v_13 + v_18)
        v_20 = self.convnext_blocks_2_dwconv(v_19)
        v_21 = self.convnext_blocks_2_norm(v_20)
        v_22 = self.convnext_blocks_2_pwconv1(v_21)
        v_23 = self.convnext_blocks_2_act(v_22)
        v_24 = self.convnext_blocks_2_pwconv2(v_23)
        v_25 = (v_19 + v_24)
        v_26 = self.convnext_blocks_3_dwconv(v_25)
        v_27 = self.convnext_blocks_3_norm(v_26)
        v_28 = self.convnext_blocks_3_pwconv1(v_27)
        v_29 = self.convnext_blocks_3_act(v_28)
        v_30 = self.convnext_blocks_3_pwconv2(v_29)
        v_31 = (v_25 + v_30)
        v_32 = self.mel_head(v_31)
        return v_32

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 192, 32, dtype=torch.float)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)
    v_2 = torch.rand(1, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2))
    mod.save("D:/project/Android_Projects/TTS/models/duration_prosody_predictor_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 192, 32, dtype=torch.float)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)
    v_2 = torch.rand(1, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1, v_2), "D:/project/Android_Projects/TTS/models/duration_prosody_predictor_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 192, 32, dtype=torch.float)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)
    v_2 = torch.rand(1, dtype=torch.float)

    import pnnx
    pnnx.export(net, "D:/project/Android_Projects/TTS/models/duration_prosody_predictor_pnnx.py.pt", (v_0, v_1, v_2))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 192, 32, dtype=torch.float)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)
    v_2 = torch.rand(1, dtype=torch.float)

    return net(v_0, v_1, v_2)

if __name__ == "__main__":
    print(test_inference())
