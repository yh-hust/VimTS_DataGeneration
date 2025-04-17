#!/bin/bash
# 存储base_dir
base_dir="/home/haoyan/data"
# 存储data_set_name
data_set_name="Hollywood_split"
# 存储clips320H路径
clips320H_dir="$base_dir/$data_set_name"
# 存储生成结果的目录
save_dir="/mnt/shanghai1-only-xuwuheng-can-fucking-use/haoyan/VimTs_output/0412"

# 指定要处理的索引（0到3）
idx=1
data_num=3
# 确保idx在合法范围内
if [ $idx -ge 0 ] && [ $idx -lt 4 ]; then
    input_dir="/mnt/shanghai1-only-xuwuheng-can-fucking-use/haoyan/VimTS/10_26_inputdata/video_folder/video_folder_0"
    export CUDA_VISIBLE_DEVICES=$idx # 指定要使用的GPU
    python gen.py --method "flowtext-based" --video_dir "$input_dir" --save_dir "$save_dir" --seed 2333
else
    echo "Invalid index (0-3): $idx"
fi
