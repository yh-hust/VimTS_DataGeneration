
from codef_function import model_init,train_process,test_process,video2image,gen_process,label_gen
from synthtext import new_synth_text_image as synth_text_image
from flow_estimate import flows_RAFT_gen
import os,shutil
def weight_space_release(video):
    weight_dir = os.path.join('ckpts', video[:-4])
    if os.path.exists(weight_dir):
        shutil.rmtree(weight_dir)
    return
def result_space_release(hparams,video):
    if os.path.exists(os.path.join(hparams.save_dir, video[:-4])):
        shutil.rmtree(os.path.join(hparams.save_dir, video[:-4]))
    weight_dir = os.path.join('ckpts', video[:-4])
    if os.path.exists(weight_dir):
        shutil.rmtree(weight_dir)

def synth_text(hparams,video_name,segtracker,grounding_caption):
    canonical_img_path = os.path.join(hparams.save_dir, video_name[:-4],'canonical_0.jpg')
    save_dir = os.path.join(hparams.save_dir, video_name[:-4])
    mask,anns,text = synth_text_image(canonical_img_path,save_dir,segtracker,grounding_caption)
    if anns == None:
        return None,None,None
    return mask,anns,text

def codef_based_gen(video,fl_model,segtracker,args):
    videos_dir = args.video_dir
    video_path = os.path.join(videos_dir,video)
    images = video2image(video_path)
    if len(images) < args.num_steps / 400:
        print(f'fews frames of {video} ')
        return
    flows, flows_confident = flows_RAFT_gen(fl_model, images)
    system, trainer = model_init(args, video, images, flows, flows_confident)
    system, trainer = train_process(args, system, trainer)
    canonical_img = test_process(args, video, trainer, images)
    for idx in range(5):
        try:
            print('train_process completed')
            grounding_caption = None
            canonical_mask,anns,text = synth_text(args,video,segtracker,grounding_caption)
            if anns == None:
                print(f'{video} no result')
                continue
            result = gen_process(args,video,canonical_mask,anns,system,trainer,images,text)
            label_gen(args,result,video)
            segtracker.restart_tracker()
            weight_space_release(video)
            break
        except:
            print(f'{video} wrong')
            weight_space_release(video)
            result_space_release(args, video)
            continue