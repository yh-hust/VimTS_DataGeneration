import os
from torchvision import transforms as T
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import torch
import glob
import torch.nn.functional as F
from raft_core.raft import RAFT
from raft_core.utils import flow_viz
from raft_core.utils.utils import InputPadder
import os
root_dir = '/home/yl/yh/yh/Video_image/'
DEVICE = 'cuda'
new_fl_args = {
        'model': './Segment_and_Track_Anything/models/raft-sintel.pth',
        #'path': hparams.root_dir,#os.path.join(root_dir, f'all_sequences/{video_name}/{video_name}'),
        #'outdir': f'{hparams.root_dir}_flow',
        'small': False,
        'mixed_precision': False,
        'if_mask': False,
        'confidence': True,
        'discrete': False,
        'thres': 4,
        #'outdir_conf': f'{hparams.root_dir}_flow_confidence',
        #'name': name,
        #'mask_dir':[f'{base_dir}/{name}/{name}_mask_0',f'{base_dir}/{name}/{name}_mask_1']
    }
class input_param(argparse.ArgumentParser):
    def __init__(self,param):
        super().__init__()
        self.model = param['model']
        self.path = param['model']
        self.outdir = param['outdir']
        self.small = param['small']
        self.mixed_precision = param['mixed_precision']
        self.if_mask = param['if_mask']
        self.confidence = param['confidence']
        self.discrete = param['discrete']
        self.thres = param['thres']
        self.outdir_conf = param['outdir_conf']
        self.name = param['name']
        if param.get('dropout') == None:
            self.dropout = 0
        else:
            self.dropout = param.get('dropout')
        if param.get('alternate_corr') == None:
            self.alternate_corr = False
        else:
            self.alternate_corr = param.get('alternate_corr')

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
def viz(img, flo,img_name=None):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imwrite(f'{img_name}', img_flo[:, :, [2,1,0]])
def new_flow_init(args=new_fl_args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args['model']))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model
def flows_RAFT_gen(model,images,args = new_fl_args):
    flows = []
    flows_confident = []
    with torch.no_grad():
        for image1, image2 in zip(images[:-1], images[1:]):
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to(DEVICE)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].to(DEVICE)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            if args.get('if_mask'):
                mask1,mask2=padder.pad(mask1.unsqueeze(0).unsqueeze(0),
                                        mask2.unsqueeze(0).unsqueeze(0))
                mask1=mask1.squeeze()
                mask2=mask2.squeeze()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low_, flow_up_ = model(image2, image1, iters=20, test_mode=True)
            flow_1to2 = flow_up.clone()
            flow_2to1 = flow_up_.clone()

            _,_,H,W=image1.shape
            x = torch.linspace(0, 1, W)
            y = torch.linspace(0, 1, H)
            grid_x,grid_y=torch.meshgrid(x,y)
            grid=torch.stack([grid_x,grid_y],dim=0).to(DEVICE)
            grid=grid.permute(0,2,1)
            grid[0]*=W
            grid[1]*=H
            if args.get('if_mask'):
                flow_up[:,:,mask1.long()==0]=10000
            grid_=grid+flow_up.squeeze()

            grid_norm=grid_.clone()
            grid_norm[0,...]=2*grid_norm[0,...]/(W-1)-1
            grid_norm[1,...]=2*grid_norm[1,...]/(H-1)-1

            flow_bilinear_=F.grid_sample(flow_up_,grid_norm.unsqueeze(0).permute(0,2,3,1),mode='bilinear',padding_mode='zeros')

            rgb_bilinear_=F.grid_sample(image2,grid_norm.unsqueeze(0).permute(0,2,3,1),mode='bilinear',padding_mode='zeros')
            rgb_np=rgb_bilinear_.squeeze().permute(1,2,0).cpu().numpy()[:, :, ::-1]
            if args.get('confidence'):
                ### Calculate confidence map using cycle consistency.
                # 1). First calculate `warped_image2` by the following formula:
                #   warped_image2 = F.grid_sample(image1, flow_2to1)
                # 2). Then calculate `warped_image1` by the following formula:
                #   warped_image1 = F.grid_sample(warped_image2, flow_1to2)
                # 3) Finally calculate the confidence map:
                #  confidence_map = metric_func(image1 - warped_image1)

                grid_2to1 = grid + flow_2to1.squeeze()
                norm_grid_2to1 = grid_2to1.clone()
                norm_grid_2to1[0, ...] = 2 * norm_grid_2to1[0, ...] / (W - 1) - 1
                norm_grid_2to1[1, ...] = 2 * norm_grid_2to1[1, ...] / (H - 1) - 1
                warped_image2 = F.grid_sample(image1, norm_grid_2to1.unsqueeze(0).permute(0,2,3,1), mode='bilinear', padding_mode='zeros')

                grid_1to2 = grid + flow_1to2.squeeze()
                norm_grid_1to2 = grid_1to2.clone()
                norm_grid_1to2[0, ...] = 2 * norm_grid_1to2[0, ...] / (W - 1) - 1
                norm_grid_1to2[1, ...] = 2 * norm_grid_1to2[1, ...] / (H - 1) - 1
                warped_image1 = F.grid_sample(warped_image2, norm_grid_1to2.unsqueeze(0).permute(0,2,3,1), mode='bilinear', padding_mode='zeros')

                error = torch.abs(image1 - warped_image1)
                confidence_map = torch.mean(error, dim=1, keepdim=True)
                confidence_map[confidence_map < args.get('thres')] = 1
                confidence_map[confidence_map >= args.get('thres')] = 0
            grid_bck=grid+flow_up.squeeze()+flow_bilinear_.squeeze()
            res=grid-grid_bck
            res=torch.norm(res,dim=0)
            mk=(res<10)&(flow_up.norm(dim=1).squeeze()>5)

            pts_src=grid[:,mk]

            pts_dst=(grid[:,mk]+flow_up.squeeze()[:,mk])

            pts_src=pts_src.permute(1,0).cpu().numpy()
            pts_dst=pts_dst.permute(1,0).cpu().numpy()
            indx=torch.randperm(pts_src.shape[0])[:30]
            # use cv2 to draw the matches in image1 and image2
            img_new=np.zeros((H,W*2,3),dtype=np.uint8)
            img_new[:,:W,:]=image1[0].permute(1,2,0).cpu().numpy()
            img_new[:,W:,:]=image2[0].permute(1,2,0).cpu().numpy()

            for j in indx:
                cv2.line(img_new,(int(pts_src[j,0]),int(pts_src[j,1])),(int(pts_dst[j,0])+W,int(pts_dst[j,1])),(0,255,0),1)

            #cv2.imwrite(f'{args.get("outdir")}/matches.png',img_new)

            #np.save(f'{args.get("outdir")}/{i:06d}.npy', flow_up.cpu().numpy())
            flows.append(flow_up.cpu().numpy())
            if args.get('confidence'):
                #np.save(f'{args.get("outdir_conf")}/{i:06d}_c.npy', confidence_map.cpu().numpy())
                flows_confident.append(confidence_map.cpu().numpy())
    return flows,flows_confident

def flow_init(args=new_fl_args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args['model']))

    model = model.module
    model.to(DEVICE)
    model.eval()
    os.makedirs(args['outdir'], exist_ok=True)
    os.makedirs(args['outdir_conf'], exist_ok=True)
    return model
def flow_gen(args,model):
    with torch.no_grad():
        images = glob.glob(os.path.join(args.get('path'), '*.png')) + \
                 glob.glob(os.path.join(args.get('path'), '*.jpg'))

        images = sorted(images)
        i=0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            if args.get('if_mask'):
                mk_file1=imfile1.split("/")
                mk_file1[-2]=f"{args.get('name')}_masks"
                mk_file1='/'.join(mk_file1)
                mk_file2=imfile2.split("/")
                mk_file2[-2]=f"{args.get('name')}_masks"
                mk_file2='/'.join(mk_file2)
                mask1=cv2.imread(mk_file1.replace('jpg','png')
                                 ,0)
                mask2=cv2.imread(mk_file2.replace('jpg','png'),
                                 0)
                mask1=torch.from_numpy(mask1).to(DEVICE).float()
                mask2=torch.from_numpy(mask2).to(DEVICE).float()
                mask1[mask1>0]=1
                mask2[mask2>0]=1
                image1*=mask1
                image2*=mask2

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            if args.get('if_mask'):
                mask1,mask2=padder.pad(mask1.unsqueeze(0).unsqueeze(0),
                                        mask2.unsqueeze(0).unsqueeze(0))
                mask1=mask1.squeeze()
                mask2=mask2.squeeze()

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low_, flow_up_ = model(image2, image1, iters=20, test_mode=True)
            flow_1to2 = flow_up.clone()
            flow_2to1 = flow_up_.clone()

            _,_,H,W=image1.shape
            x = torch.linspace(0, 1, W)
            y = torch.linspace(0, 1, H)
            grid_x,grid_y=torch.meshgrid(x,y)
            grid=torch.stack([grid_x,grid_y],dim=0).to(DEVICE)
            grid=grid.permute(0,2,1)
            grid[0]*=W
            grid[1]*=H
            if args.get('if_mask'):
                flow_up[:,:,mask1.long()==0]=10000
            grid_=grid+flow_up.squeeze()

            grid_norm=grid_.clone()
            grid_norm[0,...]=2*grid_norm[0,...]/(W-1)-1
            grid_norm[1,...]=2*grid_norm[1,...]/(H-1)-1

            flow_bilinear_=F.grid_sample(flow_up_,grid_norm.unsqueeze(0).permute(0,2,3,1),mode='bilinear',padding_mode='zeros')

            rgb_bilinear_=F.grid_sample(image2,grid_norm.unsqueeze(0).permute(0,2,3,1),mode='bilinear',padding_mode='zeros')
            rgb_np=rgb_bilinear_.squeeze().permute(1,2,0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f'{args.get("outdir")}/warped.png',rgb_np)

            if args.get('confidence'):
                ### Calculate confidence map using cycle consistency.
                # 1). First calculate `warped_image2` by the following formula:
                #   warped_image2 = F.grid_sample(image1, flow_2to1)
                # 2). Then calculate `warped_image1` by the following formula:
                #   warped_image1 = F.grid_sample(warped_image2, flow_1to2)
                # 3) Finally calculate the confidence map:
                #  confidence_map = metric_func(image1 - warped_image1)

                grid_2to1 = grid + flow_2to1.squeeze()
                norm_grid_2to1 = grid_2to1.clone()
                norm_grid_2to1[0, ...] = 2 * norm_grid_2to1[0, ...] / (W - 1) - 1
                norm_grid_2to1[1, ...] = 2 * norm_grid_2to1[1, ...] / (H - 1) - 1
                warped_image2 = F.grid_sample(image1, norm_grid_2to1.unsqueeze(0).permute(0,2,3,1), mode='bilinear', padding_mode='zeros')

                grid_1to2 = grid + flow_1to2.squeeze()
                norm_grid_1to2 = grid_1to2.clone()
                norm_grid_1to2[0, ...] = 2 * norm_grid_1to2[0, ...] / (W - 1) - 1
                norm_grid_1to2[1, ...] = 2 * norm_grid_1to2[1, ...] / (H - 1) - 1
                warped_image1 = F.grid_sample(warped_image2, norm_grid_1to2.unsqueeze(0).permute(0,2,3,1), mode='bilinear', padding_mode='zeros')

                error = torch.abs(image1 - warped_image1)
                confidence_map = torch.mean(error, dim=1, keepdim=True)
                confidence_map[confidence_map < args.get('thres')] = 1
                confidence_map[confidence_map >= args.get('thres')] = 0

            grid_bck=grid+flow_up.squeeze()+flow_bilinear_.squeeze()
            res=grid-grid_bck
            res=torch.norm(res,dim=0)
            mk=(res<10)&(flow_up.norm(dim=1).squeeze()>5)

            pts_src=grid[:,mk]

            pts_dst=(grid[:,mk]+flow_up.squeeze()[:,mk])

            pts_src=pts_src.permute(1,0).cpu().numpy()
            pts_dst=pts_dst.permute(1,0).cpu().numpy()
            indx=torch.randperm(pts_src.shape[0])[:30]
            # use cv2 to draw the matches in image1 and image2
            img_new=np.zeros((H,W*2,3),dtype=np.uint8)
            img_new[:,:W,:]=image1[0].permute(1,2,0).cpu().numpy()
            img_new[:,W:,:]=image2[0].permute(1,2,0).cpu().numpy()

            for j in indx:
                cv2.line(img_new,(int(pts_src[j,0]),int(pts_src[j,1])),(int(pts_dst[j,0])+W,int(pts_dst[j,1])),(0,255,0),1)

            cv2.imwrite(f'{args.get("outdir")}/matches.png',img_new)

            np.save(f'{args.get("outdir")}/{i:06d}.npy', flow_up.cpu().numpy())
            if args.get('confidence'):
                np.save(f'{args.get("outdir_conf")}/{i:06d}_c.npy', confidence_map.cpu().numpy())
            i += 1

            viz(image1, flow_up,f'{args.get("outdir")}/flow_up{i:03d}.png')
if __name__ == '__main__':
    video_name = 'beauty2'
    fl_args = {
        'model': './Segment_and_Track_Anything/models/raft-sintel.pth',
        'path': os.path.join(root_dir, f'all_sequences/{video_name}/{video_name}'),
        'outdir': os.path.join(root_dir,f'all_sequences/{video_name}/{video_name}_flow'),
        'small': False,
        'mixed_precision': False,
        'if_mask': False,
        'confidence': True,
        'discrete': False,
        'thres': 4,
        'outdir_conf': os.path.join(root_dir,f'all_sequences/{video_name}/{video_name}_flow_confidence'),
        'name': video_name
    }
    #args = input_param(fl_args).parse_args()
    from time import time
    t1 = time()
    model = flow_init(fl_args)
    t2 = time()
    print(t2-t1)
    flow_gen(fl_args,model)