from models.implicit_model import TranslationField
from models.implicit_model import ImplicitVideo
from models.implicit_model import ImplicitVideo_Hash
from models.implicit_model import Embedding
from models.implicit_model import AnnealedEmbedding
from models.implicit_model import AnnealedHash
from models.implicit_model import Deform_Hash3d_Warp
from codef_utils import load_ckpt
from codef_utils import get_optimizer
from codef_utils import get_scheduler
from codef_utils import get_learning_rate
from codef_utils import load_ckpt
from codef_utils import VideoVisualizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from losses import loss_dict
from losses import compute_gradient_loss
from metrics import psnr
import copy
import PIL.Image as pil
import os
import json
import cv2
import numpy as np
from einops import rearrange
from einops import repeat
from pathlib import Path
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from codef_datasets import dataset_dict
from copy import deepcopy
def mask_img_gen(img1,img2):
    # 确定两个图像的大小
    rows, cols, channels = img2.shape

    # 将第二个图像覆盖在第一个图像上
    roi = img1[0:rows, 0:cols]
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv.astype(np.uint8))
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask.astype(np.uint8))
    dst = cv2.add(img1_bg.astype(np.uint8), img2_fg.astype(np.uint8))
    img1[0:rows, 0:cols] = dst
    return img1

def mask_gen(frame,ann):
    """
    :param frame: (h,w,3)
    :param ann: (2,4,n)
    :return: frame_mask (h,w) mask in ann
    """
    h,w,_ = frame.shape
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    rect_mask = np.zeros_like(frame_gray, dtype=np.uint16)
    ann_coords = ann.reshape(2, -1)
    temp = np.zeros([ann_coords.shape[1], 2], dtype=np.int64)
    temp[:, 0] = ann_coords[0, :]
    temp[:, 1] = ann_coords[1, :]

    rect = cv2.minAreaRect(temp)
    box = cv2.boxPoints(rect).astype(np.int64)
    rect_mask = cv2.fillPoly(rect_mask, [box], 1)
    return rect_mask*(frame_gray*255)

def mask_canonical_gen(canonical,ann):
    """
    :param frame: (h,w,3)
    :param ann: (2,4,n)
    :return: mask_canonical (h,w,3) mask in ann
    """
    h,w,_ = canonical.shape
    frame_gray = cv2.cvtColor(canonical,cv2.COLOR_RGB2GRAY)
    rect_mask = np.zeros_like(frame_gray, dtype=np.uint16)
    ann_coords = ann.reshape(2, -1)
    temp = np.zeros([ann_coords.shape[1], 2], dtype=np.int64)
    temp[:, 0] = ann_coords[0, :]
    temp[:, 1] = ann_coords[1, :]

    rect = cv2.minAreaRect(temp)
    box = cv2.boxPoints(rect).astype(np.int64)
    rect_mask = cv2.fillPoly(rect_mask, [box], 1).reshape(h,w,1)
    rect_mask_3D = np.concatenate([rect_mask,rect_mask,rect_mask],axis=-1)
    return canonical * rect_mask_3D

def mask_sample(mask):
    h, w = mask.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    locs_grid = np.stack([x, y], -1)

    #kernel = np.ones((3, 3), np.uint8)
    #mask = cv2.erode(mask, kernel, 1)
    sample_mask = mask > 0
    coords = locs_grid[sample_mask]
    return coords

class ImplicitVideoSystem(LightningModule):
    def __init__(self, hparams,images,flows = None,flows_confident = None,video_name = None,anns = None,text = None):
        super(ImplicitVideoSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.color_loss = loss_dict['mse'](coef=1)
        self.images = images
        self.flows = flows
        self.flows_confident = flows_confident
        self.video_name = video_name
        self.anns = anns
        self.frames_anns = []
        self.frames = []
        self.txts = []
        self.text = text
        self.masks = []
        """
        if hparams.save_video:
            self.video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.raw_video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.dual_video_visualizer = VideoVisualizer(fps=hparams.fps)
        """
        self.models_to_train=[]
        self.embedding_xyz = Embedding(2, 8)
        self.embeddings = {'xyz': self.embedding_xyz}
        self.models = {}
        # Construct normalized meshgrid.
        h = self.hparams.img_wh[1]
        w = self.hparams.img_wh[0]
        self.img_mask = np.zeros([h,w])
        self.img_masks = []
        self.h = h
        self.w = w
        self.num_models = 1
        if hparams.encode_w:
            self.embeddings['xyz_w'] = []
            assert (isinstance(self.hparams.N_xyz_w, list))
            #print(len(self.hparams.N_xyz_w))
            in_channels_xyz = []
            for i in range(self.num_models):
                #print(self.num_models)
                N_xyz_w = self.hparams.N_xyz_w[i]
                in_channels_xyz += [2 + 2 * N_xyz_w * 2]
                if hparams.annealed:
                    if hparams.deform_hash:
                        self.embedding_hash = AnnealedHash(
                        in_channels=2,
                        annealed_step=hparams.annealed_step,
                        annealed_begin_step=hparams.annealed_begin_step)
                        self.embeddings['aneal_hash'] = self.embedding_hash
                    else:
                        self.embedding_xyz_w = AnnealedEmbedding(
                            in_channels=2,
                            N_freqs=N_xyz_w,
                            annealed_step=hparams.annealed_step,
                            annealed_begin_step=hparams.annealed_begin_step)
                        self.embeddings['xyz_w'] += [self.embedding_xyz_w]
                else:
                    self.embedding_xyz_w = Embedding(2, N_xyz_w)
                    self.embeddings['xyz_w'] += [self.embedding_xyz_w]

            for i in range(self.num_models):
                embedding_w = torch.nn.Embedding(hparams.N_vocab_w, hparams.N_w)
                torch.nn.init.uniform_(embedding_w.weight, -0.05, 0.05)
                #load_ckpt(embedding_w, hparams.weight_path, model_name=f'w_{i}')
                self.embeddings[f'w_{i}'] = embedding_w
                self.models_to_train += [self.embeddings[f'w_{i}']]

                # Add warping field mlp.
                if hparams.deform_hash:
                    with open('configs/hash.json') as f:
                        config = json.load(f)
                    warping_field = Deform_Hash3d_Warp(config=config)
                else:
                    warping_field = TranslationField(
                        D=self.hparams.deform_D,
                        W=self.hparams.deform_W,
                        in_channels_xyz=in_channels_xyz[i])

                load_ckpt(warping_field,
                          hparams.weight_path,
                          model_name=f'warping_field_{i}')
                self.models[f'warping_field_{i}'] = warping_field

        # Set up the canonical model.
        if hparams.canonical_dir is None:
            for i in range(self.num_models):
                if hparams.vid_hash:
                    with open('configs/hash.json') as f:
                        config = json.load(f)
                    implicit_video = ImplicitVideo_Hash(config=config)
                else:
                    implicit_video = ImplicitVideo(
                        D=hparams.vid_D,
                        W=hparams.vid_W,
                        sigmoid_offset=hparams.sigmoid_offset)
                load_ckpt(implicit_video, hparams.weight_path,
                          f'implicit_video_{i}')
                self.models[f'implicit_video_{i}'] = implicit_video

        for key in self.embeddings:
            setattr(self, key, self.embeddings[key])
        for key in self.models:
            setattr(self, key, self.models[key])

        self.models_to_train += [self.models]

    def deform_pts(self, ts_w, grid, encode_w, step=0, i=0):
        #pdb.set_trace()
        if self.hparams.deform_hash:
            ts_w_norm = ts_w / self.seq_len
            ts_w_norm = ts_w_norm.repeat(grid.shape[0], 1)
            input_xyt = torch.cat([grid, ts_w_norm], dim=-1)
            if 'aneal_hash' in self.embeddings.keys():
                deform = self.models[f'warping_field_{i}'](
                    input_xyt,
                    step=step,
                    aneal_func=self.embeddings['aneal_hash'])
            else:
                deform = self.models[f'warping_field_{i}'](input_xyt)
            if encode_w:
                deformed_grid = deform + grid
            else:
                deformed_grid = grid
        else:
            if encode_w:
                e_w = self.embeddings[f'w_{i}'](repeat(ts_w, 'b n ->  (b l) n ',
                                                    l=grid.shape[0])[:, 0])
                # Whether to use annealed positional encoding.
                if self.hparams.annealed:
                    pe_w = self.embeddings['xyz_w'][i](grid, step)
                else:
                    pe_w = self.embeddings['xyz_w'][i](grid)

                # Warping field type.
                deform = self.models[f'warping_field_{i}'](torch.cat(
                    [e_w, pe_w], 1))
                deformed_grid = deform + grid
            else:
                deformed_grid = grid

        return deformed_grid

    def forward(self,
                ts_w,
                grid,
                encode_w,
                step=0,
                flows=None):
        # grid -> positional encoding
        # ts_w -> embedding
        #pdb.set_trace()
        grid = rearrange(grid, 'b n c -> (b n) c')
        results_list = []
        flow_loss_list = []
        text_masks_list = []
        deform_list = []
        for i in range(self.num_models):
            deformed_grid = self.deform_pts(ts_w, grid, encode_w, step, i)  # [batch * num_pixels, 2]
            deform_list.append(deformed_grid)
            # Compute optical flow loss.
            flow_loss = 0
            if self.hparams.flow_loss > 0 and not self.hparams.test:
                if flows.max() > -1e2 and step > self.hparams.flow_step:
                    grid_new = grid + flows.squeeze(0)
                    deformed_grid_new = self.deform_pts(
                        ts_w + 1, grid_new, encode_w, step, i)
                    flow_loss = (deformed_grid_new, deformed_grid)
            flow_loss_list.append(flow_loss)
            if self.hparams.vid_hash:
                pe_deformed_grid = (deformed_grid + 0.3) / 1.6
            else:
                pe_deformed_grid = self.embeddings['xyz'](deformed_grid)
            if not self.training and self.hparams.canonical_dir is not None:
                w, h = self.img_wh
                canonical_img = self.canonical_img.squeeze(0)
                h_c, w_c = canonical_img.shape[1:3]
                grid_new = deformed_grid.clone()
                grid_new[..., 1] = (2 * deformed_grid[..., 0] - 1) * h / h_c
                grid_new[..., 0] = (2 * deformed_grid[..., 1] - 1) * w / w_c
                if len(canonical_img.shape) == 3:
                    canonical_img = canonical_img.unsqueeze(0)
                results = torch.nn.functional.grid_sample(
                    canonical_img[i:i + 1].permute(0, 3, 1, 2),
                    grid_new.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    padding_mode='border')
                results = results.squeeze().permute(1,0)
                if self.anns:
                    canonical = canonical_img[0].cpu().numpy()
                    canonical_gray = (cv2.cvtColor(canonical,cv2.COLOR_RGB2GRAY)*255).astype(np.uint32)
                    frame = rearrange(results, '(h w) c-> h w c', h=h, w=w).cpu().numpy()
                    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                    W, H = self.hparams.img_wh
                    trans_grid = grid_new.clone().cpu()
                    frame_anns = []
                    for ann in self.anns:
                        new_char_ann = np.zeros_like(ann)
                        for char_idx in range(ann.shape[-1]):
                            for k in range(4):
                                grid_x,grid_y = ann[:,k,char_idx]
                                x_norm = 2*grid_x/w_c - 1
                                y_norm = 2*grid_y/h_c - 1
                                dists = torch.sum((trans_grid-torch.tensor([x_norm,y_norm]))**2,dim=-1)
                                coords_xy = dists.argmin().cpu().numpy()
                                coords_x  = coords_xy % W
                                coords_y  = coords_xy // W
                                new_char_ann[0,k,char_idx] = coords_x
                                new_char_ann[1,k,char_idx] = coords_y
                        frame_anns.append(new_char_ann)
                    final_frame_anns = []
                    final_txt = []
                    text_masks = np.zeros_like(frame)
                    for id,ann in enumerate(frame_anns):
                        #src_mask = mask_gen(canonical,self.anns[id])
                        des_mask = mask_gen(frame,ann)
                        #index_mat = np.zeros_like(des_mask,dtype=bool)
                        d_point = mask_sample(des_mask)
                        #index_mat[d_point[:,1],d_point[:,0]] = True
                        s_point = deepcopy(d_point)
                        trans_grid_mat = rearrange(trans_grid, '(h w) c-> h w c', h=h, w=w).numpy()
                        mask_canonical = mask_canonical_gen(canonical,self.anns[id])
                        point_num = d_point.shape[0]
                        for idx in range(point_num):
                            coords_x = int(d_point[idx,0])
                            coords_y = int(d_point[idx,1])
                            #x_norm, y_norm = trans_grid[coords_x+coords_y*W]
                            x_norm,y_norm = trans_grid_mat[coords_y,coords_x]
                            s_point[idx,0] = int(((x_norm+1)/2)*w_c)
                            s_point[idx,1] = int(((y_norm+1)/2)*h_c)
                        d_point = np.concatenate((d_point,np.ones([point_num,1])), axis= -1)
                        s_point = np.concatenate((s_point,np.ones([point_num,1])), axis= -1)
                        try:
                            M = cv2.findHomography(s_point.astype('float32').copy(),d_point.astype('float32').copy(),method=0)[0].T
                            #assert check_M(M)
                            text_mask = cv2.warpPerspective(mask_canonical, M.T, (frame.shape[1], frame.shape[0]))
                            final_frame_anns.append(ann)
                            final_txt.append(self.text[id])
                            text_masks += text_mask
                        except:
                            #pdb.set_trace()
                            continue
                        #text_mask = deepcopy(canonical)

                        #text_masks += text_mask
                    text_masks_list.append(text_masks)
                    self.frames_anns.append(final_frame_anns)
                    self.txts.append(final_txt)
            else:
                results = self.models[f'implicit_video_{i}'](pe_deformed_grid)

            results_list.append(results)

        ret = edict(rgbs=results_list,
                    text_masks = text_masks_list,
                    flow_loss=flow_loss_list,
                    deform=deform_list)

        return ret

    def setup(self, stage):
        if not self.hparams.test:
            dataset = dataset_dict[self.hparams.dataset_name]
            kwargs = {
                'images':self.images,
                'flows':self.flows,
                'flows_confident':self.flows_confident,
                'img_wh': tuple(self.hparams.img_wh),
                'canonical_wh': self.hparams.canonical_wh,
                'ref_idx': self.hparams.ref_idx,
            }
            #pdb.set_trace()
            self.train_dataset = dataset(split='train', **kwargs)
            self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        lr_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [lr_dict]

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                          num_workers=0,
                          batch_size=self.hparams.batch_size,
                          sampler=sampler,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,  # validate one image (H*W rays) at a time.
            pin_memory=True)

    def test_dataloader(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'images':self.images,
            'img_wh': tuple(self.hparams.img_wh),
            'mask_dir': self.hparams.mask_dir,
            'canonical_wh': self.hparams.canonical_wh,
            'canonical_dir': self.hparams.canonical_dir,
            'test': self.hparams.test
        }
        self.train_dataset = dataset(split='train', **kwargs)
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,  # validate one image (H*W rays) at a time.
            pin_memory=True)

    def training_step(self, batch, batch_idx):
        # Fetch training data.
        #pdb.set_trace()
        rgbs = batch['rgbs']
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        flows = batch['flows']
        grid_c = batch['grid_c']
        ref_batch = batch['reference']
        self.seq_len = batch['seq_len']

        loss = 0
        rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')

        # Forward the model.
        ret = self.forward(ts_w,
                            grid,
                            self.hparams.encode_w,
                            self.global_step,
                            flows=flows)

        # Mannually set a reference frame.
        if self.hparams.ref_step < 0: self.hparams.step = 1e10
        if (self.hparams.ref_idx is not None
                and self.global_step < self.hparams.ref_step):
            rgbs_c_flattend = rearrange(ref_batch[0],
                                        'b h w c -> (b h w) c')
            ret_c = self(ts_w, grid, False, self.global_step, flows=flows)

        # Loss computation.
        for i in range(self.num_models):
            results = ret.rgbs[i]
            mk_t = rearrange(mk[i], 'b h w c -> (b h w) c')
            mk_t = mk_t.sum(dim=-1) > 0.05

            if (self.hparams.ref_idx is not None
                and self.global_step < self.hparams.ref_step):
                mk_c_t = rearrange(ref_batch[1][i], 'b h w c -> (b h w) c')
                mk_c_t = mk_c_t.sum(dim=-1) > 0.05

            # Background regularization.
            if self.hparams.bg_loss:
                mk1 = torch.logical_not(mk_t)
                if self.hparams.self_bg:
                    grid_flattened = rgbs_flattend
                else:
                    grid_flattened = rearrange(grid, 'b n c -> (b n) c')
                    grid_flattened = torch.cat(
                        [grid_flattened, grid_flattened[:, :1]], -1)

            if self.hparams.bg_loss and self.hparams.mask_dir:
                loss = loss + self.hparams.bg_loss * self.color_loss(
                    results[mk1], grid_flattened[mk1])

            # MSE color loss.
            loss = loss + self.color_loss(results[mk_t],
                                            rgbs_flattend[mk_t])

            # Image gradient loss.
            img_pred = rearrange(results,
                                 '(b h w) c -> b h w c',
                                 b=1,
                                 h=self.h,
                                 w=self.w)
            rgbs_gt = rearrange(rgbs_flattend,
                                '(b h w) c -> b h w c',
                                b=1,
                                h=self.h,
                                w=self.w)
            mk_t_re = rearrange(mk_t,
                                '(b h w c) -> b h w c',
                                b=1,
                                h=self.h,
                                w=self.w)
            grad_loss = compute_gradient_loss(rgbs_gt.permute(0, 3, 1, 2),
                                              img_pred.permute(0, 3, 1, 2),
                                              mask=mk_t_re.permute(0, 3, 1, 2))
            loss = loss + grad_loss * self.hparams.grad_loss

            # Optical flow loss.
            if ret.flow_loss[0] != 0:
                mk_flow_t = torch.logical_and(mk_t, flows[0].sum(dim=-1)< 3)
                loss = loss + torch.nn.functional.l1_loss(
                    ret.flow_loss[i][0][mk_flow_t], ret.flow_loss[i][1]
                    [mk_flow_t]) * self.hparams.flow_loss

            # Reference loss.
            if (self.hparams.ref_idx is not None
                    and self.global_step < self.hparams.ref_step):
                results_c = ret_c.rgbs[i]
                loss += self.color_loss(results_c[mk_c_t],
                                        rgbs_c_flattend[mk_c_t])

            # PSNR metric.
            with torch.no_grad():
                if i == 0:
                    psnr_ = psnr(results[mk_t], rgbs_flattend[mk_t])

        self.log('lr', get_learning_rate(self.optimizer), prog_bar=True)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgbs = batch['rgbs']
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        grid_c = grid  # batch['grid_c']
        self.seq_len = batch['seq_len']
        ret = self(ts_w, grid, self.hparams.encode_w, self.global_step)
        ret_c = self(ts_w, grid_c, False, self.global_step)

        log = {}
        W, H = self.hparams.img_wh

        rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')
        img_gt = rgbs_flattend.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        stack_list = [img_gt]
        for i in range(self.num_models):
            results = ret.rgbs[i]
            results_c = ret_c.rgbs[i]
            mk_t = rearrange(mk[i], 'b h w c -> (b h w) c')
            if batch_idx == 0:
                results[mk_t.sum(dim=-1) <= 0.05] = 0
                img = results.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                img_c = results_c.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                stack_list.append(img)
                stack_list.append(img_c)

        stack = torch.stack(stack_list) # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_Reconstructed', stack,
                                          self.global_step)

        return log

    def test_step(self, batch, batch_idx):
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        grid_c = batch['grid_c']
        W, H = self.hparams.img_wh
        self.seq_len = batch['seq_len']
        if self.hparams.canonical_dir is not None:
            self.canonical_img = batch['canonical_img']
            self.img_wh = batch['img_wh']
        """
        save_dir = os.path.join('results',
                                self.hparams.root_dir.split('/')[0],
                                self.hparams.root_dir.split('/')[1],
                                self.hparams.exp_name)
        """
        save_dir = os.path.join(self.hparams.save_dir,self.video_name)
        sample_name = self.video_name
        if self.hparams.canonical_dir is not None:
            #test_dir = f'{save_dir}_transformed'
            test_dir = save_dir
            video_name = f'{sample_name}_{self.hparams.exp_name}_transformed'
        else:
            test_dir = f'{save_dir}'
            video_name = f'{sample_name}_{self.hparams.exp_name}'
        #Path(test_dir).mkdir(parents=True, exist_ok=True)
        """
        if batch_idx > 0 and self.hparams.save_video:
            self.video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}.mp4'))
            self.raw_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_raw.mp4'))
            self.dual_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_dual.mp4'))
        """
        if batch_idx == 0 and self.hparams.canonical_dir is None:
            # Save the canonical image.
            ret = self(ts_w, grid_c, False, self.global_step)
        ret_n = self(ts_w, grid, self.hparams.encode_w, self.global_step)

        img = np.zeros((H * W, 3), dtype=np.float32)
        for i in range(self.num_models):
            if batch_idx == 0 and self.hparams.canonical_dir is None:
                results_c = ret.rgbs[i]
                if self.hparams.canonical_wh:
                    img_c = results_c.view(self.hparams.canonical_wh[1],
                                           self.hparams.canonical_wh[0],
                                           3).float().cpu().numpy()
                else:
                    img_c = results_c.view(H, W, 3).float().cpu().numpy()

                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'{test_dir}/canonical_{i}.jpg', img_c * 255)

            mk_n = rearrange(mk[i], 'b h w c -> (b h w) c')
            mk_n = mk_n.sum(dim=-1) > 0.05
            mk_n = mk_n.cpu().numpy()
            #if False:
            if self.hparams.canonical_dir is not None:
                img = ret_n.text_masks[i]

                #results = results
                # results = ret_n.rgbs[i] # (3, H, W)
                #img[mk_n] = results[mk_n]
            else:
                results = ret_n.rgbs[i]
                results = results.cpu().numpy()
                #results = ret_n.rgbs[i] # (3, H, W)
                img[mk_n] = results[mk_n]
                img = rearrange(img, '(h w) c -> h w c', h=H, w=W)
        img = img * 255
        img = img[..., ::-1].astype(np.uint8)
        rgbs = batch['rgbs'].view(H, W, 3).cpu().numpy() * 255
        rgbs = rgbs.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        if self.hparams.canonical_dir is not None:
            self.masks.append(img)
            img_with_text = mask_img_gen(deepcopy(rgbs), deepcopy(img))
            img_with_text =cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)
            gen_img = img_with_text
            #gen_img = cv2.morphologyEx(gen_img,cv2.MORPH_OPEN,kernel)
            #cv2.imwrite(f'{test_dir}/{batch_idx:05d}.png', gen_img)
            self.frames.append(gen_img)
        else:
            img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
            #cv2.imwrite(f'{test_dir}/{batch_idx:05d}.png', img)
            gen_img = img
        #gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        """
        if batch_idx > 0 and self.hparams.save_video:
            self.video_visualizer.add(gen_img)
            self.raw_video_visualizer.add(rgbs)
            dual_img = np.concatenate((rgbs, gen_img), axis=1)
            self.dual_video_visualizer.add(dual_img)
        """




        if self.hparams.save_deform:
            save_deform_dir = f'{test_dir}_deform'
            Path(save_deform_dir).mkdir(parents=True, exist_ok=True)
            deformation_field = ret_n.deform[0]
            deformation_field = rearrange(deformation_field,
                                          '(h w) c -> h w c', h=H, w=W)
            grid_ = rearrange(grid[0], '(h w) c -> h w c', h=H, w=W)
            deformation_delta = deformation_field - grid_
            np.save(f'{save_deform_dir}/{batch_idx:05d}.npy',
                    deformation_delta.cpu().numpy())

    def on_test_epoch_end(self):
        """
        if self.hparams.save_video:
            self.video_visualizer.save()
            self.raw_video_visualizer.save()
            self.dual_video_visualizer.save()
        """
        return

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items