
from Codef_SynthText.synthgen import RendererV10
import time
import random
import pickle
import os
import numpy as np
import cv2
from codef_function import read_img,depth_model_init,depthset_es,seg_model_init,segset_es,depth_es,seg_es
import PIL.Image as pil
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import depth.networks as networks
SEED = 16
np.random.seed(SEED)
random.seed(SEED)
SECS_PER_IMG = 5
INSTANCE_PER_IMAGE = 1
np.random.seed()

def save_pkl(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)
    return
def get_from_pkl(path):
    with open(path,'rb') as f:
        imgs_dir,image_set,depth_set,seg_set,area_set,label_set = pickle.load(f)
    return imgs_dir,image_set,depth_set,seg_set,area_set,label_set
def get_data(img_path):
    #datadir = DATA_PATH
    files = os.listdir(img_path)
    imgs_dir = [os.path.join(img_path,file) for file in files]
    depth_model = depth_model_init()
    image_set = read_img(img_path)
    depth_set = depthset_es(image_set,depth_model)
    #depth_set = []
    seg_model =seg_model_init()
    seg_set,area_set,label_set = segset_es(image_set,seg_model)
    return imgs_dir,image_set,depth_set,seg_set,area_set,label_set
def pipeline(image,depth,seg,area,label,RV10):

    #txt = None
    for i in range(10):
        res = RV10.render_text(image, depth, seg, area, label)
        if len(res) !=0:
            break
    #text_masks =
    #print(f"synth text:{res[0]['txt']}")
    #plt.imshow(res[0]['img'])
    #plt.show()
    #res.im_final, text, bb, collision_mask
    return res
def synth_text(imagedir,outputdir=None):
    time1 = time.time()
    datadir = DATA_PATH
    RV10 = RendererV10(datadir, max_time=SECS_PER_IMG)
    #imgs_dir,image_set,depth_set, seg_set, area_set, label_set = get_data(imagedir)
    #imgs_dir, image_set, depth_set, seg_set, area_set, label_set = get_from_pkl('test_data.pkl')
    imgs_dir, image_set, depth_set, seg_set, area_set, label_set = get_from_pkl('temp_data.pkl')
    time2 = time.time()
    print(f'loading data and init: {time2-time1}')
    results = []
    for i in range(len(image_set)):
        time1 = time.time()
        result = pipeline(image_set[i],depth_set[i],seg_set[i],area_set[i],label_set[i],RV10)
        time2 = time.time()
        print(f"process_one_image_time:{time2-time1}")
        #print(i)
        results.append(result)
    if outputdir != None:
        save_dir = os.path.join(outputdir,f'canonical_img.png')
        #img = result['img']
        mask = result['img_with_text']
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_dir,mask)

def get_image_info(image_path,segtracker,grounding_caption):
    imgs_dir = image_path
    depth_model = depth_model_init()
    image = pil.open(image_path).convert('RGB')
    depth = depth_es(image, depth_model)
    #seg_model = seg_model_init()

    H, W = image.size
    seg_img = np.zeros([W, H])
    image = np.array(image)
    box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True
    if grounding_caption == None:
        pred_mask = segtracker.seg(image)
    else:
        pred_mask, _ = segtracker.detect_and_seg(image, grounding_caption, box_threshold, text_threshold,
                                             box_size_threshold, reset_image)
    label = np.unique(pred_mask)
    label = label[label!=0]
    #sam_result = seg_es(image, seg_model)
    #label = np.array([i + 1 for i in range(len(sam_result))])
    area = []
    for idx in range(len(label)):
        area.append(np.sum(np.where(pred_mask==label[idx],1,0)))
    seg_img = pred_mask
    #import pdb;pdb.set_trace()
    return image, depth, seg_img, np.array(area), label


def new_synth_text_image(image_path,outputdir,segtracker,grounding_caption):
    #save_dir = os.path.join(outputdir, f'canonical_img.png')
    #image = cv2.imread(save_dir)
    #cv2.imwrite(save_dir, mask)
    #return
    time1 = time.time()
    datadir = DATA_PATH
    RV10 = RendererV10(datadir, max_time=SECS_PER_IMG)
    image, depth, seg, area, label = get_image_info(image_path,segtracker,grounding_caption)
    result = pipeline(image, depth, seg, area, label, RV10)
    save_path = os.path.join(outputdir, f'canonical_img.jpg')
    if outputdir != None:
        if len(result) ==0:
            cv2.imwrite(save_path, cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            print('synthtext no result')
            return None,None,None
        else:
            #save_dir = os.path.join(outputdir, f'canonical_img.jpg')
            img = result['img']
            mask = result['text_mask']
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            anns = result['wordBB']
            text = result['txt']
            cv2.imwrite(save_path, mask)
            #cv2.imwrite(f'/home/yl/yh/yh/Video_image/canonical_img/{time1}.jpg',img)
    time2 = time.time()
    print(f"synthtext cost {time2-time1}")
    return mask,anns,text

IMAGE_PATH = '/mnt/shanghai1-only-xuwuheng-can-fucking-use/haoyan/VimTS/VTD/VimTs_output/0412/0JfJGE1AZrg_0/canonical_0.jpg'
TXT_PATH = 'pipeline_data/txt/'
IMAGE_INFO_PATH='pipeline_data/image_info/'
OUTPUT_PATH = 'pipeline_output/result/'
DATA_PATH = 'data'
if __name__=='__main__':
    import argparse
    import random
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    parser.add_argument('--imagedir', type=str, default=IMAGE_PATH)
    parser.add_argument('--image_info', type=str, default=IMAGE_INFO_PATH)
    parser.add_argument('--txtdir', type=str, default=TXT_PATH)
    parser.add_argument('--outputdir', type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    from Segment_and_Track_Anything.SegTracker import SegTracker
    from params import params
    segtracker = SegTracker(params['segtracker']['segtracker_args'], params['segtracker']['sam_args'], params['segtracker']['aot_args'])
    seed = int(16)
    np.random.seed(seed)
    random.seed(seed)
    new_synth_text_image(args.imagedir,args.outputdir,segtracker,None)