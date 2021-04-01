# Super Resolution With ESRGAN and DeepMux

[*ESRGAN*](https://github.com/xinntao/ESRGAN) is a SoTA model in image super resolution. Here we show how to create a functional interface for the model and deploy it to DeepMux platform.

## Setting up the environment.

To run everything, you need to install *deepmux-cli* and log in. This is done with the following two commands

```
pip install deepmux-cli
deepmux login 
```

The last command will ask you to enter your unique API token, which you can find on app.deepmux.com.

Let's clone the repository:

`git clone https://github.com/xinntao/ESRGAN.git && cd ESRGAN`

## Downloading pretrained weights

Download the files from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) and put them into `./models` folder.

## Writing some code:

The original testing code is contained within `test.py`, where the model iterates over a specific folder and outputs resulting images into another folder. We want something different - to take bytes as an input and output bytes as well.

So, let's create a `func_esrgan.py` file and write the following code:

```python
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# print('Model path {:s}. \nTesting...'.format(model_path))


def superres(data):
    path = 'lr.png'
    
    with open(path, 'wb') as f:
        f.write(data)
    
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('hr.png', output)
    
    with open('hr.png', 'rb') as f:
        output_bytes = f.read()
        
    os.remove('lr.png')
    os.remove('hr.png')
    
    return output_bytes
```

The `model_path` variable may be changed to use another one of the provided models. See the [original repo](https://github.com/xinntao/ESRGAN) for more details

## Adding requirements and initializing:

Create a `requirements.txt` file. As we are going to launch the model in a pytorch-cuda environment, we only need to specify *opencv* here:
```
opencv-python
```
Our project is almost ready! We write the following command:

`deepmux init`

The deepmux.yaml file will appear in the project folder. Fill it in with your data:

```yaml
name: sentiment_analysis # project name
env: <...>
python:
  call: main:classify # file and function to call
  requirements: requirements.txt # path to requirements file
```

To fill in the env line, ru the deepmux env command. The result will be something like this:
```name: python3.6 language: python
name: python3.7 language: python
name: python3.6-tensorflow2.1-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow2.1-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow2.2-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow1.13.1-pytorch-1.3-cuda10.0 language: python
name: python3.7-mmdetection-pytorch-1.6-cuda10.1 language: python
```
As we do not need tensorflow, let's choose the last environment.

The resulting yaml file should look like this:
```yaml
name: esrgan
env: python3.7-mmdetection-pytorch-1.6-cuda10.1
python:
  call: func_esrgan:superres
  requirements: requirements.txt

```
## Loading the model:

Your model is ready to deploy! Now, just call the `deepmux upload` command. This might take a few minutes to process.

## Running the model:

The model is uploaded and ready to use. Let's run the model on an image!

To do this, you need to run the following command:

```shell
curl -X POST -H "X-Token: <YOUR TOKEN>" https://api.deepmux.com/v1/function/esrgan/run --data-binary "@baboon.png" > baboon_hr.png
```
and model's output will be saved to `baboon_hr.png`.
