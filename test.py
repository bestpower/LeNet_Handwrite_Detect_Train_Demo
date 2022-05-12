#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import onnx
import onnxruntime
from mnist import LeNet

def pytorch2Onnx(device, input, input_pytorch_model_path, output_onnx_model_path):
    # load pytorch model
    net = LeNet().to(device)
    net.load_state_dict(torch.load(input_pytorch_model_path, map_location=device))
    net.eval()
    # An example input you would normally provide to your model's forward() method
    input_names = ['data']
    output_names = ['prob']
    # Export the onnx model
    torch.onnx._export(net, input, output_onnx_model_path,
                       export_params=True, verbose=True, input_names=input_names, output_names=output_names)

def pytorchModelTest(device, input, pytorch_model_path):
    # load pytorch model
    torch_model = LeNet().to(device)
    torch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    # set the model to inference mode
    torch_model.eval()
    # input to the model
    test_torch_out = torch_model(input)
    print("torch out:", test_torch_out)
    return test_torch_out

def onnxModelCheck(sim_onnx_model_path):
    # load onnx model
    onnx_model = onnx.load(sim_onnx_model_path)
    onnx.checker.check_model(onnx_model)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnxModelTest(input, sim_onnx_model_path):
    # load onnx model
    ort_session = onnxruntime.InferenceSession(sim_onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("onnx out:", ort_outs[0])
    return ort_outs

def compareOnnx_Pytorch(test_torch_out, test_onnx_out):
    # compare onnx runtime and pyTorch results
    np.testing.assert_allclose(to_numpy(test_torch_out), test_onnx_out[0], rtol=1e-03, atol=1e-03)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_pytorch_model_path = "./Models/30/LeNet_p27.pth"
    output_onnx_model_path = "./Models/LeNet_p27.onnx"
    sim_onnx_model_path = "./Models/LeNet_p27_sim.onnx"

    # 随机生成数据对比测试
    # input = torch.randn(1, 1, 28, 28, device=device)
    # # pytorch->onnx
    # pytorch2Onnx(device, input, input_pytorch_model_path, output_onnx_model_path)
    # # onnx->sim_onnx
    # # pytorch model test
    # test_torch_out = pytorchModelTest(device, input, input_pytorch_model_path)
    # # simple onnx model check
    # onnxModelCheck(sim_onnx_model_path)
    # # simple onnx model test
    # test_onnx_out = onnxModelTest(input, sim_onnx_model_path)
    # # compare onnx runtime and pytorch results
    # compareOnnx_Pytorch(test_torch_out, test_onnx_out)

    # 指定数据对比测试
    # image_path = "./data/TestDigitImgs/1600417425118.jpg" # 1 -> 7
    # image_path = "./data/TestDigitImgs/1600417457655.jpg" # 2 -> 2
    # image_path = "./data/TestDigitImgs/1600417437655.jpg" # 1 -> 4
    image_path = "./data/TestDigitImgs/1600417463500.jpg" # 0 -> 2
    # image_path = "./data/TestDigitImgs/1600417480123.jpg" # 0 -> 2

    img = Image.open(image_path).convert('L')
    test_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])
    img2 = test_transform(img)
    img2 = torch.unsqueeze(img2, 0)
    img2 = img2.cuda().float() if torch.cuda.is_available() else img2.cpu().float()
    # pytorch test
    test_torch_out = pytorchModelTest(device, img2, input_pytorch_model_path)
    _, torch_predicted = torch.max(test_torch_out, 1)
    print("pytorch_test_img " + image_path + " out = ", torch_predicted)
    # onnx test
    test_onnx_out = onnxModelTest(img2, sim_onnx_model_path)
    _, onnx_predicted = torch.max(torch.from_numpy(test_onnx_out[0]), 1)
    print("onnx_test_img " + image_path + " out = ", onnx_predicted)
