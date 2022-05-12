# LeNet_Handwrite_Detect_Train_Demo

## LeNet手写数字识别Pytorch训练及测试示例详解

### 0. run environment

> os: ubuntu 18.04

> python version: 3.6+

> torch 1.2+

> torchvision 0.4+

> opencv-python 

> numpy 

> onnx 

> onnxruntime

### 1.Train and save pytorch model file each epoch

'''

    $ python3 mnist.py
'''

### 2.Find and select the highest accuracy epoch saved model according to the train log

### 3.Convert pytorch model to onnx model and compare their results

#### 3.1 Convert pytorch model to onnx model

'''

    $ sudo pip3 install onnx-simplifier
    $ python3 -m onnxsim ./Models/LeNet_p27.onnx ./Models/LeNet_p27_sim.onnx
    $ python3 onnx_test.py
'''

#### 3.2 Test pytorch model and get result

##### 3.2.1 Input data is random abstract data

'''python

    # load pytorch model
    torch_model = LeNet().to(device)
    torch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    # set the model to inference mode
    torch_model.eval()
    # input to the model
    test_torch_out = torch_model(input)
    # print inference result
    print("torch out:", test_torch_out)
'''

##### 3.2.2 Input data is selected data file

'''

    # read img data
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
'''

#### 3.3 Test onnx model and get result

##### 3.3.1 Input data is random abstract data

'''python

    # load onnx model
    ort_session = onnxruntime.InferenceSession(sim_onnx_model_path)
    # set input format
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    # input to the model
    ort_outs = ort_session.run(None, ort_inputs)
    # print inference result
    print("onnx out:", ort_outs)
'''

##### 3.3.2 Input data is selected data file

'''python

    img = Image.open(image_path).convert('L')
    test_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])
    img2 = test_transform(img)
    img2 = torch.unsqueeze(img2, 0)
    img2 = img2.cuda().float() if torch.cuda.is_available() else img2.cpu().float()
    # onnx test
    test_onnx_out = onnxModelTest(img2, sim_onnx_model_path)
    _, onnx_predicted = torch.max(torch.from_numpy(test_onnx_out[0]), 1)
    print("onnx_test_img " + image_path + " out = ", onnx_predicted)
'''

#### 3.4 Compare pytorch and onnx model test results

'''python

    np.testing.assert_allclose(to_numpy(test_torch_out), test_onnx_out[0], rtol=1e-03, atol=1e-05)
'''
