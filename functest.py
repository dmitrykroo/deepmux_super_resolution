from func_esrgan import superres

with open('LR/baboon.png', 'rb') as f:
    data = f.read()
    
data_hr = superres(data)

with open('baboon_hr.png', 'wb') as f:
    f.write(data_hr)