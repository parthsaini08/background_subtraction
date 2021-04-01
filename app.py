import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Flask, render_template, request,send_from_directory
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config['UPLOAD_PATH']='demo/image_matting/colab/foreground/'
@app.route('/')
def home():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        import os
        os.getcwd()
        import sys
        import argparse
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        from src.models.modnet import MODNet
        # %cd /content
        #if not os.path.exists('MODNet'):
         # !git clone https://github.com/ZHKKKe/MODNet
        # %cd MODNet/
        #pretrained_ckpt = 'MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
        #if not os.path.exists(pretrained_ckpt):
         # !gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz \
          #        -O pretrained/modnet_photographic_portrait_matting.ckpt
        
        #from google.colab import drive
        #drive.mount('/content/drive')

        import shutil
        
        input_folder = 'demo/image_matting/colab/input'
        lists=os.listdir(input_folder)
        for l in lists:
            os.remove(os.path.join(input_folder,l))
        
        output_folder = 'demo/image_matting/colab/output'
        lists=os.listdir(output_folder)
        for l in lists:
            os.remove(os.path.join(output_folder,l))
        
        foreground_folder = 'demo/image_matting/colab/foreground'
        lists=os.listdir(foreground_folder)
        for l in lists:
            os.remove(os.path.join(foreground_folder,l))
        
        # upload images (PNG or JPG)
        shutil.move('./{}'.format(f.filename),'demo/image_matting/colab/input')
        
        #image_names = list(files.upload().keys())
        #for image_name in image_names:
         # shutil.move(image_name, os.path.join(input_folder, image_name))
         
        '''!python -m demo.image_matting.colab.inference \
                --input-path demo/image_matting/colab/input \
                --output-path demo/image_matting/colab/output \
                --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt'''
        
        input_path='demo/image_matting/colab/input'
        output_path='demo/image_matting/colab/output'
        ckpt_path='pretrained/modnet_photographic_portrait_matting.ckpt'


        #using inference.py
        # define hyper-parameters
        ref_size = 512
        
            # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        
            # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet).cuda()
        modnet.load_state_dict(torch.load(ckpt_path))
        modnet.eval()

            # inference images
        im_names = os.listdir(input_path)
        for im_name in im_names:
            print('Process image: {0}'.format(im_name))
            im = Image.open(os.path.join(input_path, im_name))
        
                # unify image channels to 3
            im = np.asarray(im)
            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]
        
                # convert image to PyTorch tensor
            im = Image.fromarray(im)
            im = im_transform(im)
        
                # add mini-batch dim
            im = im[None, :, :, :]
        
                # resize image for input
            im_b, im_c, im_h, im_w = im.shape
            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    im_rh = ref_size
                    im_rw = int(im_w / im_h * ref_size)
                elif im_w < im_h:
                    im_rw = ref_size
                    im_rh = int(im_h / im_w * ref_size)
            else:
                im_rh = im_h
                im_rw = im_w
                
            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        
                # inference
            _, _, matte = modnet(im.cuda(), True)
        
                # resize and save matte
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            matte_name = im_name.split('.')[0] + '.png'
            Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))
        

        
        import numpy as np
        from PIL import Image
        import cv2
        import numpy as np
        from skimage import io
        def combined_display(image, matte):
          w, h = image.width, image.height
          rw, rh = 800, int(h * 800 / (3 * w))
          image = np.asarray(image)
          if len(image.shape) == 2:
            image = image[:, :, None]
          if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
          elif image.shape[2] == 4:
            image = image[:, :, 0:3]
          matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
          foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
          
          combined = np.concatenate((image, foreground, matte * 255), axis=1)
          foreground = Image.fromarray(np.uint8(foreground)).resize((w, h))
          return foreground


        def notcombined_display(image, matte):
        
          w, h = image.width, image.height
          rw, rh = 800, int(h * 800 / (3 * w))
        
          image = np.asarray(image)
          if len(image.shape) == 2:
            image = image[:, :, None]
          if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
          elif image.shape[2] == 4:
            image = image[:, :, 0:3]
          matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
          foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
          
          combined = np.concatenate((image, foreground, matte * 255), axis=1)
          combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
          return image,matte,foreground
          
        image_names = os.listdir(input_folder)
        for image_name in image_names:
          matte_name = image_name.split('.')[0] + '.png'
          image = Image.open(os.path.join(input_folder, image_name))
          matte = Image.open(os.path.join(output_folder, matte_name))
          matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
          bitnd = image*matte+np.full(np.asarray(image).shape, 255) * (1 - matte)
          print(bitnd.shape)
          bitnd = Image.fromarray(np.uint8(bitnd))
          bitnd.save(f'demo/image_matting/colab/foreground/{image_name}')
          print(image_name, '\n')
        
        image_name = os.listdir(foreground_folder)
        return render_template('newview.html', filename=image_name[0])
                		
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)
if __name__ == '__main__':
   app.run(debug = True)
