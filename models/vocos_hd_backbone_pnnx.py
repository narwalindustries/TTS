# pnnx model stat
# model inputshape = [1,80,64]f32
# FLOPS = 1.763G
# memory OPS = 19.316M

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

        self.in_proj = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=80, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.norm = nn.GroupNorm(affine=True, eps=1.000000e-5, num_channels=512, num_groups=1)
        self.backbone_0_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=512, in_channels=512, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.backbone_0_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=512, num_groups=1)
        self.backbone_0_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=2048, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_0_act = nn.GELU()
        self.backbone_0_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=2048, kernel_size=(1,), out_channels=512, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_1_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=512, in_channels=512, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.backbone_1_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=512, num_groups=1)
        self.backbone_1_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=2048, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_1_act = nn.GELU()
        self.backbone_1_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=2048, kernel_size=(1,), out_channels=512, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_2_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=512, in_channels=512, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.backbone_2_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=512, num_groups=1)
        self.backbone_2_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=2048, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_2_act = nn.GELU()
        self.backbone_2_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=2048, kernel_size=(1,), out_channels=512, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_3_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=512, in_channels=512, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.backbone_3_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=512, num_groups=1)
        self.backbone_3_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=2048, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_3_act = nn.GELU()
        self.backbone_3_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=2048, kernel_size=(1,), out_channels=512, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_4_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=512, in_channels=512, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.backbone_4_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=512, num_groups=1)
        self.backbone_4_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=2048, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_4_act = nn.GELU()
        self.backbone_4_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=2048, kernel_size=(1,), out_channels=512, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_5_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=512, in_channels=512, kernel_size=(7,), out_channels=512, padding=(3,), padding_mode='zeros', stride=(1,))
        self.backbone_5_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=512, num_groups=1)
        self.backbone_5_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=2048, padding=(0,), padding_mode='zeros', stride=(1,))
        self.backbone_5_act = nn.GELU()
        self.backbone_5_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=2048, kernel_size=(1,), out_channels=512, padding=(0,), padding_mode='zeros', stride=(1,))
        self.mag_head = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=513, padding=(0,), padding_mode='zeros', stride=(1,))
        self.phase_head = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=512, kernel_size=(1,), out_channels=1026, padding=(0,), padding_mode='zeros', stride=(1,))

        archive = zipfile.ZipFile('D:/project/Android_Projects/TTS/models/vocos_hd_backbone.pnnx.bin', 'r')
        self.in_proj.bias = self.load_pnnx_bin_as_parameter(archive, 'in_proj.bias', (512), 'float32')
        self.in_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'in_proj.weight', (512,80,7), 'float32')
        self.norm.bias = self.load_pnnx_bin_as_parameter(archive, 'norm.bias', (512), 'float32')
        self.norm.weight = self.load_pnnx_bin_as_parameter(archive, 'norm.weight', (512), 'float32')
        self.backbone_0_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.dwconv.bias', (512), 'float32')
        self.backbone_0_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.dwconv.weight', (512,1,7), 'float32')
        self.backbone_0_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.norm.bias', (512), 'float32')
        self.backbone_0_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.norm.weight', (512), 'float32')
        self.backbone_0_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.pwconv1.bias', (2048), 'float32')
        self.backbone_0_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.pwconv1.weight', (2048,512,1), 'float32')
        self.backbone_0_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.pwconv2.bias', (512), 'float32')
        self.backbone_0_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.0.pwconv2.weight', (512,2048,1), 'float32')
        self.backbone_1_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.dwconv.bias', (512), 'float32')
        self.backbone_1_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.dwconv.weight', (512,1,7), 'float32')
        self.backbone_1_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.norm.bias', (512), 'float32')
        self.backbone_1_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.norm.weight', (512), 'float32')
        self.backbone_1_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.pwconv1.bias', (2048), 'float32')
        self.backbone_1_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.pwconv1.weight', (2048,512,1), 'float32')
        self.backbone_1_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.pwconv2.bias', (512), 'float32')
        self.backbone_1_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.1.pwconv2.weight', (512,2048,1), 'float32')
        self.backbone_2_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.dwconv.bias', (512), 'float32')
        self.backbone_2_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.dwconv.weight', (512,1,7), 'float32')
        self.backbone_2_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.norm.bias', (512), 'float32')
        self.backbone_2_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.norm.weight', (512), 'float32')
        self.backbone_2_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.pwconv1.bias', (2048), 'float32')
        self.backbone_2_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.pwconv1.weight', (2048,512,1), 'float32')
        self.backbone_2_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.pwconv2.bias', (512), 'float32')
        self.backbone_2_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.2.pwconv2.weight', (512,2048,1), 'float32')
        self.backbone_3_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.dwconv.bias', (512), 'float32')
        self.backbone_3_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.dwconv.weight', (512,1,7), 'float32')
        self.backbone_3_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.norm.bias', (512), 'float32')
        self.backbone_3_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.norm.weight', (512), 'float32')
        self.backbone_3_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.pwconv1.bias', (2048), 'float32')
        self.backbone_3_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.pwconv1.weight', (2048,512,1), 'float32')
        self.backbone_3_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.pwconv2.bias', (512), 'float32')
        self.backbone_3_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.3.pwconv2.weight', (512,2048,1), 'float32')
        self.backbone_4_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.dwconv.bias', (512), 'float32')
        self.backbone_4_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.dwconv.weight', (512,1,7), 'float32')
        self.backbone_4_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.norm.bias', (512), 'float32')
        self.backbone_4_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.norm.weight', (512), 'float32')
        self.backbone_4_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.pwconv1.bias', (2048), 'float32')
        self.backbone_4_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.pwconv1.weight', (2048,512,1), 'float32')
        self.backbone_4_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.pwconv2.bias', (512), 'float32')
        self.backbone_4_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.4.pwconv2.weight', (512,2048,1), 'float32')
        self.backbone_5_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.dwconv.bias', (512), 'float32')
        self.backbone_5_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.dwconv.weight', (512,1,7), 'float32')
        self.backbone_5_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.norm.bias', (512), 'float32')
        self.backbone_5_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.norm.weight', (512), 'float32')
        self.backbone_5_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.pwconv1.bias', (2048), 'float32')
        self.backbone_5_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.pwconv1.weight', (2048,512,1), 'float32')
        self.backbone_5_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.pwconv2.bias', (512), 'float32')
        self.backbone_5_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'backbone.5.pwconv2.weight', (512,2048,1), 'float32')
        self.mag_head.bias = self.load_pnnx_bin_as_parameter(archive, 'mag_head.bias', (513), 'float32')
        self.mag_head.weight = self.load_pnnx_bin_as_parameter(archive, 'mag_head.weight', (513,512,1), 'float32')
        self.phase_head.bias = self.load_pnnx_bin_as_parameter(archive, 'phase_head.bias', (1026), 'float32')
        self.phase_head.weight = self.load_pnnx_bin_as_parameter(archive, 'phase_head.weight', (1026,512,1), 'float32')
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

    def forward(self, v_0):
        v_1 = self.in_proj(v_0)
        v_2 = self.norm(v_1)
        v_3 = self.backbone_0_dwconv(v_2)
        v_4 = self.backbone_0_norm(v_3)
        v_5 = self.backbone_0_pwconv1(v_4)
        v_6 = self.backbone_0_act(v_5)
        v_7 = self.backbone_0_pwconv2(v_6)
        v_8 = (v_2 + v_7)
        v_9 = self.backbone_1_dwconv(v_8)
        v_10 = self.backbone_1_norm(v_9)
        v_11 = self.backbone_1_pwconv1(v_10)
        v_12 = self.backbone_1_act(v_11)
        v_13 = self.backbone_1_pwconv2(v_12)
        v_14 = (v_8 + v_13)
        v_15 = self.backbone_2_dwconv(v_14)
        v_16 = self.backbone_2_norm(v_15)
        v_17 = self.backbone_2_pwconv1(v_16)
        v_18 = self.backbone_2_act(v_17)
        v_19 = self.backbone_2_pwconv2(v_18)
        v_20 = (v_14 + v_19)
        v_21 = self.backbone_3_dwconv(v_20)
        v_22 = self.backbone_3_norm(v_21)
        v_23 = self.backbone_3_pwconv1(v_22)
        v_24 = self.backbone_3_act(v_23)
        v_25 = self.backbone_3_pwconv2(v_24)
        v_26 = (v_20 + v_25)
        v_27 = self.backbone_4_dwconv(v_26)
        v_28 = self.backbone_4_norm(v_27)
        v_29 = self.backbone_4_pwconv1(v_28)
        v_30 = self.backbone_4_act(v_29)
        v_31 = self.backbone_4_pwconv2(v_30)
        v_32 = (v_26 + v_31)
        v_33 = self.backbone_5_dwconv(v_32)
        v_34 = self.backbone_5_norm(v_33)
        v_35 = self.backbone_5_pwconv1(v_34)
        v_36 = self.backbone_5_act(v_35)
        v_37 = self.backbone_5_pwconv2(v_36)
        v_38 = (v_32 + v_37)
        v_39 = self.mag_head(v_38)
        v_40 = torch.clamp(v_39, max=10.0, min=-12.0)
        v_41 = torch.exp(v_40)
        v_42 = self.phase_head(v_38)
        v_43, v_44 = torch.tensor_split(v_42, dim=1, indices=(513,))
        v_45 = torch.sqrt(((torch.pow(v_43, 2) + torch.pow(v_44, 2)) + 1.000000e-8))
        v_46 = (v_43 / v_45)
        v_47 = (v_44 / v_45)
        v_48 = (v_41, v_46, v_47, )
        return v_48

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 80, 64, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("D:/project/Android_Projects/TTS/models/vocos_hd_backbone_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 80, 64, dtype=torch.float)

    torch.onnx.export(net, v_0, "D:/project/Android_Projects/TTS/models/vocos_hd_backbone_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 80, 64, dtype=torch.float)

    import pnnx
    pnnx.export(net, "D:/project/Android_Projects/TTS/models/vocos_hd_backbone_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 80, 64, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
