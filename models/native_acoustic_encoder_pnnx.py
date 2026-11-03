# pnnx model stat
# model inputshape = [1,32]i64,[1]i64
# FLOPS = 79.589M
# memory OPS = 1.88M

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

        self.phoneme_emb = nn.Embedding(embedding_dim=192, num_embeddings=512, sparse=False)
        self.lang_emb = nn.Embedding(embedding_dim=192, num_embeddings=32, sparse=False)
        self.encoder_blocks_0_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=192, in_channels=192, kernel_size=(7,), out_channels=192, padding=(3,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_0_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=192, num_groups=1)
        self.encoder_blocks_0_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=192, kernel_size=(1,), out_channels=768, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_0_act = nn.GELU()
        self.encoder_blocks_0_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=768, kernel_size=(1,), out_channels=192, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_1_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=192, in_channels=192, kernel_size=(7,), out_channels=192, padding=(3,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_1_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=192, num_groups=1)
        self.encoder_blocks_1_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=192, kernel_size=(1,), out_channels=768, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_1_act = nn.GELU()
        self.encoder_blocks_1_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=768, kernel_size=(1,), out_channels=192, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_2_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=192, in_channels=192, kernel_size=(7,), out_channels=192, padding=(3,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_2_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=192, num_groups=1)
        self.encoder_blocks_2_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=192, kernel_size=(1,), out_channels=768, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_2_act = nn.GELU()
        self.encoder_blocks_2_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=768, kernel_size=(1,), out_channels=192, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_3_dwconv = nn.Conv1d(bias=True, dilation=(1,), groups=192, in_channels=192, kernel_size=(7,), out_channels=192, padding=(3,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_3_norm = nn.GroupNorm(affine=True, eps=1.000000e-6, num_channels=192, num_groups=1)
        self.encoder_blocks_3_pwconv1 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=192, kernel_size=(1,), out_channels=768, padding=(0,), padding_mode='zeros', stride=(1,))
        self.encoder_blocks_3_act = nn.GELU()
        self.encoder_blocks_3_pwconv2 = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=768, kernel_size=(1,), out_channels=192, padding=(0,), padding_mode='zeros', stride=(1,))
        self.proj = nn.Conv1d(bias=True, dilation=(1,), groups=1, in_channels=192, kernel_size=(1,), out_channels=192, padding=(0,), padding_mode='zeros', stride=(1,))

        archive = zipfile.ZipFile('D:/project/Android_Projects/TTS/models/native_acoustic_encoder.pnnx.bin', 'r')
        self.phoneme_emb.weight = self.load_pnnx_bin_as_parameter(archive, 'phoneme_emb.weight', (512,192), 'float32')
        self.lang_emb.weight = self.load_pnnx_bin_as_parameter(archive, 'lang_emb.weight', (32,192), 'float32')
        self.encoder_blocks_0_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.dwconv.bias', (192), 'float32')
        self.encoder_blocks_0_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.dwconv.weight', (192,1,7), 'float32')
        self.encoder_blocks_0_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.norm.bias', (192), 'float32')
        self.encoder_blocks_0_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.norm.weight', (192), 'float32')
        self.encoder_blocks_0_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.pwconv1.bias', (768), 'float32')
        self.encoder_blocks_0_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.pwconv1.weight', (768,192,1), 'float32')
        self.encoder_blocks_0_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.pwconv2.bias', (192), 'float32')
        self.encoder_blocks_0_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.0.pwconv2.weight', (192,768,1), 'float32')
        self.encoder_blocks_1_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.dwconv.bias', (192), 'float32')
        self.encoder_blocks_1_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.dwconv.weight', (192,1,7), 'float32')
        self.encoder_blocks_1_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.norm.bias', (192), 'float32')
        self.encoder_blocks_1_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.norm.weight', (192), 'float32')
        self.encoder_blocks_1_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.pwconv1.bias', (768), 'float32')
        self.encoder_blocks_1_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.pwconv1.weight', (768,192,1), 'float32')
        self.encoder_blocks_1_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.pwconv2.bias', (192), 'float32')
        self.encoder_blocks_1_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.1.pwconv2.weight', (192,768,1), 'float32')
        self.encoder_blocks_2_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.dwconv.bias', (192), 'float32')
        self.encoder_blocks_2_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.dwconv.weight', (192,1,7), 'float32')
        self.encoder_blocks_2_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.norm.bias', (192), 'float32')
        self.encoder_blocks_2_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.norm.weight', (192), 'float32')
        self.encoder_blocks_2_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.pwconv1.bias', (768), 'float32')
        self.encoder_blocks_2_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.pwconv1.weight', (768,192,1), 'float32')
        self.encoder_blocks_2_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.pwconv2.bias', (192), 'float32')
        self.encoder_blocks_2_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.2.pwconv2.weight', (192,768,1), 'float32')
        self.encoder_blocks_3_dwconv.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.dwconv.bias', (192), 'float32')
        self.encoder_blocks_3_dwconv.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.dwconv.weight', (192,1,7), 'float32')
        self.encoder_blocks_3_norm.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.norm.bias', (192), 'float32')
        self.encoder_blocks_3_norm.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.norm.weight', (192), 'float32')
        self.encoder_blocks_3_pwconv1.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.pwconv1.bias', (768), 'float32')
        self.encoder_blocks_3_pwconv1.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.pwconv1.weight', (768,192,1), 'float32')
        self.encoder_blocks_3_pwconv2.bias = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.pwconv2.bias', (192), 'float32')
        self.encoder_blocks_3_pwconv2.weight = self.load_pnnx_bin_as_parameter(archive, 'encoder_blocks.3.pwconv2.weight', (192,768,1), 'float32')
        self.proj.bias = self.load_pnnx_bin_as_parameter(archive, 'proj.bias', (192), 'float32')
        self.proj.weight = self.load_pnnx_bin_as_parameter(archive, 'proj.weight', (192,192,1), 'float32')
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

    def forward(self, v_0, v_1):
        v_2 = self.phoneme_emb(v_0)
        v_3 = self.lang_emb(v_1)
        v_4 = v_3.unsqueeze(1)
        v_5 = (v_2 + v_4)
        v_6 = torch.transpose(v_5, dim0=1, dim1=2)
        v_7 = self.encoder_blocks_0_dwconv(v_6)
        v_8 = self.encoder_blocks_0_norm(v_7)
        v_9 = self.encoder_blocks_0_pwconv1(v_8)
        v_10 = self.encoder_blocks_0_act(v_9)
        v_11 = self.encoder_blocks_0_pwconv2(v_10)
        v_12 = (v_6 + v_11)
        v_13 = self.encoder_blocks_1_dwconv(v_12)
        v_14 = self.encoder_blocks_1_norm(v_13)
        v_15 = self.encoder_blocks_1_pwconv1(v_14)
        v_16 = self.encoder_blocks_1_act(v_15)
        v_17 = self.encoder_blocks_1_pwconv2(v_16)
        v_18 = (v_12 + v_17)
        v_19 = self.encoder_blocks_2_dwconv(v_18)
        v_20 = self.encoder_blocks_2_norm(v_19)
        v_21 = self.encoder_blocks_2_pwconv1(v_20)
        v_22 = self.encoder_blocks_2_act(v_21)
        v_23 = self.encoder_blocks_2_pwconv2(v_22)
        v_24 = (v_18 + v_23)
        v_25 = self.encoder_blocks_3_dwconv(v_24)
        v_26 = self.encoder_blocks_3_norm(v_25)
        v_27 = self.encoder_blocks_3_pwconv1(v_26)
        v_28 = self.encoder_blocks_3_act(v_27)
        v_29 = self.encoder_blocks_3_pwconv2(v_28)
        v_30 = (v_24 + v_29)
        v_31 = self.proj(v_30)
        return v_31

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, 32), dtype=torch.long)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)

    mod = torch.jit.trace(net, (v_0, v_1))
    mod.save("D:/project/Android_Projects/TTS/models/native_acoustic_encoder_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, 32), dtype=torch.long)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)

    torch.onnx.export(net, (v_0, v_1), "D:/project/Android_Projects/TTS/models/native_acoustic_encoder_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, 32), dtype=torch.long)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)

    import pnnx
    pnnx.export(net, "D:/project/Android_Projects/TTS/models/native_acoustic_encoder_pnnx.py.pt", (v_0, v_1))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, 32), dtype=torch.long)
    v_1 = torch.randint(10, (1, ), dtype=torch.long)

    return net(v_0, v_1)

if __name__ == "__main__":
    print(test_inference())
