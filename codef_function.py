import copy

import PIL.Image as pil
import os
import warnings
import sys
sys.path.append('src/groundingdino/')
warnings.filterwarnings("ignore")
import json
import cv2
import numpy as np
from einops import rearrange
from einops import repeat
from pathlib import Path
from easydict import EasyDict as edict
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy
from losses import loss_dict
from losses import compute_gradient_loss

from models.implicit_model import ImplicitVideo

from codef_utils import get_optimizer
from codef_utils import get_scheduler
from codef_utils import get_learning_rate
from codef_utils import load_ckpt
from codef_utils import VideoVisualizer

#from opt import get_opts
#from test_opt import get_opts
#from opt_train import get_opts
from all_opt import get_opts,change_opts
from metrics import psnr
from copy import deepcopy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import itertools
import matplotlib.pyplot as plt
import random
from ImplicitVideoSystem import ImplicitVideoSystem
from all_opt import change_opts
import depth.networks as networks
from params import params
from depth.layers import disp_to_depth

def vis_sample(path = 'tmp/sample',pltann=True, is_video=True,plt_char_ann = True):

    if is_video:
        images = sorted([os.path.join(path,el) for el in os.listdir(path) if el.endswith('jpg') and el.startswith('000')])
        h, w = cv2.imread(images[0]).shape[:2]
        save_path = os.path.join(path,'viz_ann.mp4') if pltann else os.path.join(path,'viz.mp4')
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
        with open(os.path.join(path,'ann.json'), 'r') as fr:
            anns = json.load(fr)
        for image,ann in zip(images,anns):
            img = cv2.imread(image)
            if pltann:
                lst = list(ann['instances'].values())
                for tmp in lst:
                    bbox = tmp['coords']
                    text = tmp['text']
                    if bbox == None:
                        continue
                    cv2.polylines(img,[np.array(bbox).astype(np.int32)],True,color=[0,255,0],thickness=2)
                    #cv2.putText(img,text,np.array(bbox).mean(0).astype(np.int32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
            if plt_char_ann:
                idx = 0
                lst = list(ann['instances'].values())
                for tmp in lst:
                    cbox = tmp['char_coords']
                    bbox = tmp['coords']
                    text = tmp['text']
                    chars = tmp['chars']
                    if cbox == None:
                        continue
                    char_num = len(cbox)
                    for i in range(char_num):
                        cbox_i = cbox[i]
                        cv2.polylines(img, [np.array(cbox_i).astype(np.int32)], True, color=[0, 255, 255], thickness=2)
                        try:
                            cv2.putText(img,chars[i],np.array(cbox_i).mean(0).astype(np.int32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
                        except:
                            print(i)
                    idx +=1
            # plt.imshow(img)

            # plt.show()
            video.write(img)
        video.release()
    else:
        save_path = os.path.join(path, 'viz_ann.jpg') if pltann else os.path.join(path, 'viz.jpg')
        image = [os.path.join(path,el) for el in os.listdir(path) if el.endswith('jpg')][0]
        with open(os.path.join(path,'ann.json'), 'r') as fr:
            ann = json.load(fr)[0]
        img = cv2.imread(image)
        if pltann:
            lst = list(ann['instances'].values())
            for tmp in lst:
                bbox = tmp['coords']
                text = tmp['text']
                cv2.polylines(img, [np.array(bbox).astype(np.int32)], True, color=[0, 255, 0], thickness=2)
                cv2.putText(img, text, np.array(bbox).mean(0).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
        cv2.imwrite(save_path,img)

def coord_2_list(coord):
    j,k,l = coord.shape
    if j != 2 or k != 4:
        return None
    char_coord = []
    for k in range(l):
        temp_list = []
        for i in range(4):
            temp_list.append([int(coord[:,i,k][0]),int(coord[:,i,k][1])])
        char_coord.append(temp_list)
    return char_coord

def model_init(hparams,video_name,images,flows,flows_confident):
    os.makedirs(hparams.save_dir,exist_ok=True)
    os.makedirs('ckpts',exist_ok=True)
    os.makedirs('logs',exist_ok=True)
    video_save_dir = os.path.join(hparams.save_dir,video_name[:-4])
    video_save_ckpt = os.path.join('ckpts',video_name[:-4])
    video_save_log = os.path.join('logs',video_name[:-4])
    os.makedirs(f'{video_save_dir}',exist_ok=True)
    os.makedirs(f'{video_save_ckpt}', exist_ok=True)
    os.makedirs(f'{video_save_log}', exist_ok=True)

    #os.makedirs( f'{os.path.join(hparams.output_dir,video_name,'logs')}',exist_ok=True)
    base_dir,name =f'all_sequences',hparams.root_dir.split('/')[-1]
    H,W,_ = images[0].shape
    #H = W = 590
    setparams_dict = {
        'test':False,
        'canonical_dir':None,
        'model_save_path': video_save_ckpt,
        'log_save_path': video_save_log,
        'encode_w': True,
        'annealed':True,
        'config':f"configs/base.yaml",
        'flow_dir':f'{hparams.root_dir}_flow',
        'img_wh':[W,H],
        'canonical_wh':[W,H],
        'weight_path':None,
        #'canonical_wh':[int(1.1*W)//10*10,int(1.1*H)//10*10],
        'dataset_name':'videos'
    }
    hparams = change_opts(setparams_dict, hparams)
    system = ImplicitVideoSystem(hparams,images,flows,flows_confident,video_name[:-4])
    checkpoint_callback = ModelCheckpoint(
        dirpath=video_save_ckpt,
        filename='{step:d}',
        mode='max',
        save_top_k=0,
        every_n_train_steps=hparams.save_model_iters,
        save_last=True)
    logger = TensorBoardLogger(save_dir=hparams.log_save_path,
                               name=hparams.exp_name)
    trainer = Trainer(max_steps=hparams.num_steps,
                      #max_epochs=1,
                      precision=16 if hparams.vid_hash == True else 32,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      accelerator='gpu',
                      devices=hparams.gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      #profiler="simple" if len(hparams.gpus) == 1 else None,
                      profiler=None,
                      val_check_interval=hparams.valid_iters,
                      limit_val_batches=hparams.valid_batches,
                      #limit_train_batches = 40,
                      strategy="ddp_find_unused_parameters_true")
    return system,trainer

def train_process(hparams,system,trainer):
    #pdb.set_trace()
    """
    setparams_dict = {
        'model_save_path': f'ckpts/{base_dir}/{name}',
        'log_save_path': f'logs/{base_dir}/{name}',
        #'mask_dir':None,
        'encode_w': True,
        'annealed':True,
        'config':f"configs/base.yaml",
        'flow_dir':f'{hparams.root_dir}_flow',
        'img_wh':[W,H],
        #'canonical_wh':[W,H],
        'canonical_wh':[int(W*1.1)//10*10,int(H*1.1)//10*10],
        #'mask_dir':None
        #'mask_dir':[f'{base_dir}/{name}/{name}_mask_0',f'{base_dir}/{name}/{name}_mask_1']
        #'mask_dir':[f'{base_dir}/{name}/{name}_mask_0'],
    }
    """
    #hparams = change_opts(setparams_dict, hparams)
    trainer.fit(system)
    return system,trainer

def test_process(hparams,video_name,trainer,images):
    video_save_ckpt = os.path.join('ckpts', video_name[:-4])
    train2test_dict = {
        'test': True,
        'encode_w': True,
        'flow_dir': None,
        'save_deform': False,
        'weight_path': f'{video_save_ckpt}/last.ckpt',
        'annealed': False,
        # 'mask_dir':[f'{base_dir}/{name}/{name}_mask_0']
    }
    hparams = change_opts(train2test_dict, hparams)
    system = ImplicitVideoSystem(hparams,images,None,None,video_name[:-4])
    trainer.test(system, dataloaders=system.test_dataloader())
    save_dir = os.path.join(hparams.save_dir, video_name[:-4])
    canonical_img = cv2.imread(f'{save_dir}/canonical_0.jpg')
    canonical_img = cv2.cvtColor(canonical_img, cv2.COLOR_BGR2RGB)
    return canonical_img

def gen_process(hparams,video_name,canonical_mask,anns,system,trainer,images,text):
    video_save_ckpt = os.path.join('ckpts', video_name[:-4])
    canonical_dir = os.path.join(hparams.save_dir, video_name[:-4],'canonical_img.jpg')
    test2canonical_dict = {
        'test': True,
        'encode_w': True,
        'canonical_dir': canonical_dir,
        'weight_path': f'{video_save_ckpt}/last.ckpt',
        'save_deform': False,
    }
    hparams = change_opts(test2canonical_dict, hparams)
    system = ImplicitVideoSystem(hparams, images, None, None, video_name[:-4], anns,text)

    trainer.test(system, dataloaders=system.test_dataloader())

    return system

def char2wordBB(charBB, text):
    """
    Converts character bounding-boxes to word-level
    bounding-boxes.

    charBB : 2x4xn matrix of BB coordinates
    text   : the text string

    output : 2x4xm matrix of BB coordinates,
                 where, m == number of words.
    """
    wrds = text.split()
    bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
    wordBB = np.zeros((2, 4, len(wrds)), 'float32')

    for i in range(len(wrds)):
        cc = charBB[:, :, bb_idx[i]:bb_idx[i + 1]]

            # fit a rotated-rectangle:
            # change shape from 2x4xn_i -> (4*n_i)x2
        cc = np.squeeze(np.concatenate(np.dsplit(cc, cc.shape[-1]), axis=1)).T.astype('float32')
        rect = cv2.minAreaRect(cc.copy())
        box = np.array(cv2.boxPoints(rect))

            # find the permutation of box-coordinates which
            # are "aligned" appropriately with the character-bb.
            # (exhaustive search over all possible assignments):
        cc_tblr = np.c_[cc[0, :],
                        cc[-3, :],
                        cc[-2, :],
                        cc[3, :]].T
        perm4 = np.array(list(itertools.permutations(np.arange(4))))
        dists = []
        for pidx in range(perm4.shape[0]):
            d = np.sum(np.linalg.norm(box[perm4[pidx], :] - cc_tblr, axis=1))
            dists.append(d)
        wordBB[:, :, i] = box[perm4[np.argmin(dists)], :].T
    wordbb = []
    for i in range(4):
        wordbb.append([int(wordBB[:,i,0][0]),int(wordBB[:,i,0][1])])
    return wordbb


def next_wbb(mask, wbb):
    next_wbbs = []
    for wbb_ in wbb:
        wbb_mask = cv2.fillPoly(np.zeros_like(mask), [wbb_.astype(np.int64)], 1)
        mask_ = wbb_mask
        mask_ = mask_
        next_wbb = self.mask2wbb(mask_)
        next_wbbs.append(next_wbb)

    return next_wbbs

def label_gen(hparams,system,video_name):
    texts = system.txts
    frames = system.frames
    frames_anns = system.frames_anns
    save_dir = os.path.join(hparams.save_dir,video_name[:-4])
    anns = []
    key_dir = f'{hparams.save_dir}/{video_name[:-4]}/{len(frames)//2:06}.jpg'
    for i in range(len(frames)):
        cv2.imwrite(os.path.join(save_dir,f'{i:06}.jpg'),frames[i])
    for i in range(len(frames)):
        ann = dict()
        words = dict()
        frame_ann = frames_anns[i]
        chars = []
        coords = []
        instances = dict()
        img_dir = f'{hparams.save_dir}/{video_name[:-4]}/{i:06}.jpg'
        text = texts[i]
        for j in range(len(text)):
            words[j] = text[j]
            chars.append(list(text[j]))
            instance = dict()
            char_coords = []
            instance['text'] = text[j]
            instance['coords'] = char2wordBB(frame_ann[j],text[j])
            #coords = char2wordBB(frame_ann[j],text[j]).astype(np.uint64)
            instance['char_coords'] = coord_2_list(frame_ann[j])
            instance['chars'] = chars[j]
            instances[j] = instance
        ann['img'] = img_dir
        ann['words'] = words
        ann['chars'] = chars
        ann['key'] = key_dir
        ann['instances'] =  instances
        anns.append(ann)
    #save_ann = os.path.join(save, 'ann.json')
    with open(os.path.join(save_dir, 'ann.json'), 'w') as fw:
        json.dump(anns, fw)
    vis_sample(save_dir, pltann=False, plt_char_ann=False)
    vis_sample(save_dir, pltann=True, plt_char_ann=True)
    return

def video2image(video_path):
    #os.makedirs(image_save_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(frame)
        #filename = f"{frame_idx:0>6d}.png"
        frame_idx += 1
        #if frame_idx > 20:
            #break
    cap.release()
    print(f"frame_num:{frame_idx}")
    return images

def read_img(path,rg = (1280,1280)):
    imgs_dir = os.listdir(path)
    imgs = []
    for img_dir in imgs_dir:
        file_path = os.path.join(path,img_dir)
        img = np.array(Image.open(file_path))
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.array(T.Resize((rg[0],rg[1]))(pil.fromarray(img)))
        imgs.append(img)

def depth_model_init():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    encoder_path = os.path.join("data/ckpts/depth", "encoder.pth")
    depth_decoder_path = os.path.join("data/ckpts/depth", "depth.pth")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    depth_model = [encoder, depth_decoder, feed_height, feed_width]
    return depth_model

def depthset_es(img_set,depth_model):
    depth_imgs = []
    for img in img_set:
        depth = depth_es(img,depth_model)
        depth_imgs.append(depth)
    return depth_imgs

def seg_model_init():
    sam_args = {
        'sam_checkpoint': "./download_weight/sam_ckpt/sam_vit_h_4b8939.pth",
        'model_type': "vit_h",
        #'sam_checkpoint': "./download_weight/sam_ckpt/sam_vit_b_01ec64.pth",
        #'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 16,
            #'points_per_batch': 512,
            'pred_iou_thresh': 0.86,
            'stability_score_thresh': 0.95,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area':200,
        },
    }
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    mask_generator = SamAutomaticMaskGenerator(sam, **sam_args['generator_args'])
    #mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def segset_es(img_set,seg_model):
    seg_imgs = []
    area_set = []
    label_set = []
    for image in img_set:
        H, W, _ = image.shape
        seg_img = np.zeros([H,W])
        sam_result = seg_es(image,seg_model)
        label_set.append(np.array([i+1 for i in range(len(sam_result))]))
        area = []
        sam_result = sorted(sam_result,key=(lambda x:x['area']),reverse=True)
        for idx in range(len(sam_result)):
            area.append(sam_result[idx]['area'])
            seg_img[sam_result[idx]['segmentation']] = idx+1
        seg_imgs.append(seg_img)
        area_set.append(np.array(area))
    return seg_imgs,area_set,label_set

def depth_es(input_image, model,rg=(0.1,100)):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    encoder = model[0]
    depth_decoder = model[1]
    depths = []
    with torch.no_grad():
        feed_height = model[2]
        feed_width = model[3]
        original_width, original_height = input_image.size
        #input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            #output_name = os.path.splitext(os.path.basename(image_path))[0]
        rg = params['depth']['range']
        scaled_disp, depth = disp_to_depth(disp_resized, rg[0], rg[1])
        depths.append(depth.cpu().numpy()[0][0])
        if len(depths)==1:
            return depths[0]
        else:
            return depths

def seg_es(image,model):
    seg_img = model.generate(image)
    return seg_img