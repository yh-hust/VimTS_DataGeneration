import torch
import depth.networks as networks
import random
from synthgen import *
from flow_functions import depth_es, mixin_segmentation_es, flow_es, video2images, vis_sample,flow_args,RAFTGMA
from flow_functions import VisualizationDemo,video_segmentation_args,vVisualizationDemo,setup_cfg,vsetup_cfg,segmentation_args
import os
import traceback
import shutil
import numpy as np
from shapely.geometry import Polygon
import time,warnings
from PIL import Image
import json
warnings.filterwarnings('ignore')
meta = {
    'SECS_PER_IMG': 5,  # max time per image in seconds
    'DATA_PATH': 'data',
    'ntry': 1,
    'NUM_REP': 3,
}
def model_init():
    # flow_es init
    fl_args = flow_args()
    flow_model = torch.nn.DataParallel(RAFTGMA(fl_args))
    flow_model.load_state_dict(torch.load(fl_args.model))
    # deep_es init
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
    # seg init
    args = segmentation_args()
    cfg = setup_cfg(args)
    demo1 = VisualizationDemo(cfg)
    args = video_segmentation_args()
    cfg = vsetup_cfg(args)
    demo2 = vVisualizationDemo(cfg=cfg)
    seg_model = [demo1, demo2]
    return flow_model,depth_model,seg_model

def flowtext_based_gen(video_path, rg, save,flow_model,depth_model,seg_model):
    try:
        save_ann = os.path.join(save,'ann.json')
        if os.path.exists(save_ann):
            print(f'{save} already exists')
            return
        t1 = time.time()
        # video to image list
        if os.path.exists(save):
            shutil.rmtree(save)
        video2images(video_path, path=save, range=rg)
        images = sorted([os.path.join(save, el) for el in os.listdir(save)])

        # random sample a key frame the paint the init text
        # key = 4
        key = len(images)//2
        # key = random.randint(1, len(images) - 2)
    except:
        traceback.print_exc()
        print(f'fail to generate for {video_path} in range {str(rg)}')
        if os.path.exists(save):
            shutil.rmtree(save)
        return

    try:
        t = 0
        while t < 5:
            #seg_model.restart_tracker()
            print('model inference ...')
            # flow,depth,segmentation estimation

            RV10 = RendererV10(meta['DATA_PATH'], max_time=meta['SECS_PER_IMG'])
            imgs = [np.array(Image.open(el)) for el in images]
            flows = flow_es(images,flow_model)
            depthk = depth_es(images[key],depth_model)
            depths = [depthk]*len(imgs)
            segs, areas, labels = mixin_segmentation_es(images, key,seg_model)
            

            print('text painting')
            # paint text for all frames
            res = RV10.render_text(imgs, flows, depths, segs, areas, labels, key)

            if len(res) == 0:
                t = t + 1
                continue
            else:
                break

        # no valid paint
        if len(res) == 0:
            print(f'no results.')
            noresult_txt_dir = './noresult.txt'
            with open(noresult_txt_dir,'a',encoding='utf-8') as file:
                file.write(video_path)
            if os.path.exists(save):
                shutil.rmtree(save)
            return

        # write painted images
        out_imgs = [el['img'] for el in res]
        #import pdb;pdb.set_trace()
        for image, img in zip(images, out_imgs):
            cv2.imwrite(image, img[:,:,::-1])
        h, w, _ = img.shape


        # write annotations
        anns = []
        ids = list(range(len(res[0]['wordBB'])))
        #txts = ' '.join(res[0]['txt']).split()
        txts = res[0]['txt']
        i = 0
        for res_, image in zip(res, images):
            ann = dict()
            ann['charBB'] = res_['charBB']
            # ann['char'] = dict
            ann['wordBB'] = res_['wordBB']
            ann['words'] = dict(zip(ids, res_['txt']))
            ann['chars'] = []
            for j in range(len(res_['txt'])):
                ann['chars'].append(list(res_['txt'])[j])
            # ann['chars'] = dict(zip(ids,))
            instances = dict()
            final_chars = []
            for id, txt in zip(ids, txts):
                try:
                    box = ann['wordBB'][id]
                    cbox = ann['charBB'][id]
                    char = ann['chars'][id]
                except:
                    print(id)
                box = np.array(box).astype(np.int32)
                cbox = np.array(cbox).astype(np.int32)
                m = cbox.shape[0]  # m/4 is char num
                if m > 2:
                    try:
                        cbox = cbox.reshape(int(m / 4), 4, 2)
                    except:
                        return

                try:
                    mask = Polygon(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.int32))
                    poly = Polygon(np.array(box).astype(np.int32))
                    npoly = poly.intersection(mask)

                    hw = cv2.minAreaRect(box)[1]
                    minhw = min(hw)
                    asp = max(hw) / (min(hw) + 1e-6)
                    if npoly.area < 10 or minhw < 5:
                        continue
                    coords = box.tolist()
                    char_coords = cbox.tolist()
                    final_char_coords = []
                    final_char = []
                    char = char.replace(' ', '')
                    char = char.replace('\n','')
                    for c, coord in zip(char, char_coords):
                        charx = np.array(coord)[:, 0]
                        chary = np.array(coord)[:, 1]
                        threshold = 20
                        if (charx+threshold>0).all() and (charx-threshold<w).all() and (chary+threshold>0).all() and (chary-threshold<h).all():
                            final_char.append(c)
                            final_char_coords.append(coord)
                        else:
                            continue
                    final_chars.append(final_char)
                    instances[id] = {'text': txt, 'coords': coords, 'char_coords': final_char_coords,
                                     'chars': final_char}
                except:
                    continue
            ann['instances'] = instances
            ann['chars'] = final_chars
            ann['img'] = image
            ann['key'] = images[key]
            del ann['wordBB']
            del ann['charBB']
            anns.append(ann)
            i = i + 1

        save_ann = os.path.join(save, 'ann.json')
        with open(save_ann, 'w') as fw:
            json.dump(anns, fw)
        t2 = time.time()
        print('successful paint a clip with {:.2f}s'.format(t2 - t1))
        vis_sample(save, pltann=False, plt_char_ann=False)
        vis_sample(save, pltann=True)
    except:
        traceback.print_exc()
        print(f'fail to paint.')
        if os.path.exists(save):
            shutil.rmtree(save)
        print(images)

    return