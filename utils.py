from groundingdino.util.inference import load_model, load_image, predict
import cv2
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
from torchvision.ops import box_convert 
import warnings
import cv2
import torch
import numpy as np
from PIL import Image
import os
# 忽略所有的警告
warnings.filterwarnings("ignore")   
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

def draw_mask_and_box_on_image(image, mask, box, label):
    # 将 mask 转为 uint8 类型, 方便后续处理
    mask = mask.astype(np.uint8)
    
    # 创建一个红色的图像，用于显示 mask
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask * 255  # 只填充红色通道（BGR格式）

    # 将原始图像与红色 mask 叠加
    image_with_mask = cv2.addWeighted(image, 1.0, red_mask, 0.5, 0)
    
    # 确保边框坐标为整数
    x1, y1, x2, y2 = map(int, box)
    
    # 画出边框 (box), 颜色为绿色
    cv2.rectangle(image_with_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框，厚度为 2
    
    # 在框的左上角绘制 label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # 字体大小
    font_thickness = 2  # 字体粗细
    text_color = (0, 255, 0)  # 字体颜色 (绿色)
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = y1 - 10  # 使文本位置略微偏离框的上边缘

    # 绘制文本
    cv2.putText(image_with_mask, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return image_with_mask

def load_sam_img(image_path):

    image = cv2.imread(image_path)  # 读取的图像以NumPy数组的形式存储在变量image中
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR颜色空间转换为RGB颜色空间，还原图片色彩（图像处理库所认同的格式）
    return image

def load_GD_model(config_path, model_path):
    model = load_model(config_path, model_path)
    return model
def load_sam_model(sam_checkpoint, model_type):
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device="cuda:0")
    predictor = SamPredictor(sam_model)
    return predictor

def get_boxes(image_path, prompts, model):
    _, image = load_image(image_path)
    prompt_list = prompts.split(".")
    boxes_list = []
    logits_list = []
    phrases_list = []
    for prompt in prompt_list:
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        for box in boxes:
            boxes_list.append(box.tolist())
        for logit in logits:
            logits_list.append(logit.tolist())
        for phrase in phrases:
            phrases_list.append(phrase)
        
    return boxes_list, logits_list, phrases_list
    # annotated_frame = annotate(image_source=annotated_frame, boxes=boxes, logits=logits, phrases=phrases)#繪畫
    # cv2.imwrite(save_viusal_path, annotated_frame)#繪畫
    
def get_masks(image_path, prompts):
    #得到box
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_path = "GroundingDINO/groundingdino_swint_ogc.pth"
    GD_model = load_GD_model(config_path, model_path)
    print("GD_model loaded")
    boxes_list, logits_list, phrases_list = get_boxes(image_path, prompts, GD_model)
        # 完成后释放 GD_model
    del GD_model  # 删除模型对象
    torch.cuda.empty_cache()  # 清空CUDA缓存 (如果使用GPU)

    print("GD_model released")
    #得到mask
    sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"  # 定义模型路径
    model_type = "vit_h"  # 定义模型类型
    predictor = load_sam_model(sam_checkpoint, model_type)
    print("Sam_model loaded")
    image = load_sam_img(image_path)
    if len(boxes_list) == 0:
        return Image.fromarray(image)
    h, w, _ = image.shape
    for i in range(len(boxes_list)):
        coordinates = torch.tensor(boxes_list[i])
        label = phrases_list[i]
        box = coordinates * torch.Tensor([w, h, w, h])  # Scale the coordinates to the image size
        input_box = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False, 
        )

        mask = masks[0]  # 获取第一个 mask
        
        # 将 mask 绘制到原图上
        image = draw_mask_and_box_on_image(image, mask,input_box,label)
        
        # 你可以选择保存或显示这个带有 mask 的图像

        
    del predictor  # 删除模型对象
    torch.cuda.empty_cache()  # 清空CUDA缓存 (如果使用GPU)
    print("Sam_model released")
    return Image.fromarray(image)

def process_image(image_path, prompts):

    mask = get_masks(image_path, prompts)
    return mask

if __name__ == '__main__':
    image_dir = "image/"
    save_dir = "output/"
    image_path = os.listdir(image_dir)
    for image_path in image_path:
        image_path = os.path.join(image_dir, image_path)
        image_name = image_path.split("/")[-1]
        prompts = image_name[:-4]
        if os.path.exists(os.path.join(save_dir, image_name)):
            continue
        image = get_masks(image_path, prompts)
        print(type(image))
        # 保存图像到本地
        save_path = "output_image_with_mask.png"  # 可以修改为你想保存的路径和文件名
        # 将 PIL 图像转换为 NumPy 数组
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR颜色空间转换为RGB颜色空间，还原图片色彩（图像处理库所认同的格式）
        
        cv2.imwrite(os.path.join(save_dir, image_name), image)