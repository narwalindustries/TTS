import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.randint(10, (1, 32), dtype=torch.int)
    in1 = torch.randint(10, (1, ), dtype=torch.int)
    out = []

    with ncnn.Net() as net:
        net.load_param("D:/project/Android_Projects/TTS/models/native_acoustic_encoder.ncnn.param")
        net.load_model("D:/project/Android_Projects/TTS/models/native_acoustic_encoder.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy(), batch_index=0).clone())
            ex.input("in1", ncnn.Mat(in1.numpy(), batch_index=233).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(out0.numpy(batch_index=0)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
