import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils import process_image

# 初始化全局变量
image1_path = None

def upload_image1():
    """上传图片并显示"""
    global image1_path
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        
        # 调整图片大小，保持最大尺寸为512x512
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # 创建Tkinter兼容格式
        img_tk = ImageTk.PhotoImage(img)
        
        # 更新图片显示
        label_img1.config(image=img_tk)
        label_img1.image = img_tk  # 保存引用

        # 存储图片路径以便后续处理
        image1_path = file_path

def process_and_display():
    """处理图像并显示结果"""
    global image1_path
    if image1_path:
        # 从文本框获取用户输入
        prompt = entry_text.get()
        processed_img = process_image(image1_path, prompt)  # 传递文本到处理函数

        processed_img = processed_img.convert('RGB')
        # 确保处理后的图片大小为512x512
        processed_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter图像格式
        img_tk = ImageTk.PhotoImage(processed_img)
        
        # 更新处理后的图片显示
        label_img2.config(image=img_tk)
        label_img2.image = img_tk  # 保存引用

def exit_program():
    """退出程序"""
    root.quit()
    root.destroy()

def setup_window():
    """设置窗口的大小、位置及其他基本属性"""
    root = tk.Tk()
    root.title("GroundingDino_SAM_Process_Image")

    # 设置窗口大小并居中显示
    window_width = 1200
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 计算窗口的初始位置，使其居中
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 设置窗口的最小尺寸
    root.minsize(600, 400)

    return root

def setup_widgets(root):
    """设置窗口中的所有组件"""
    # 添加退出按钮
    button_exit = tk.Button(root, text="Exit", command=exit_program, width=10)
    button_exit.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

    # 图片展示框1
    label_img1 = tk.Label(root, text="Upload Image 1")
    label_img1.grid(row=0, column=0, padx=10, pady=10)

    # 图片展示框2
    label_img2 = tk.Label(root, text="Processed Image")
    label_img2.grid(row=0, column=1, padx=10, pady=10)

    # 上传按钮
    button_upload = tk.Button(root, text="Upload Image", command=upload_image1, width=20)
    button_upload.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

    # 文本输入框
    entry_text = tk.Entry(root, width=20)
    entry_text.grid(row=1, column=1, pady=10, padx=10, sticky="ew")
    entry_text.insert(0, "Enter text here")  # 提示默认文本

    # 处理按钮
    button_process = tk.Button(root, text="Process Image", command=process_and_display, width=20)
    button_process.grid(row=2, column=0, columnspan=2, pady=20, padx=10, sticky="ew")

    # 配置列和行的权重，确保组件的大小可以调整
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # 将 `label_img1` 和 `label_img2` 引用传递到函数中
    return label_img1, label_img2, entry_text

def main():
    """主函数"""
    global label_img1, label_img2, entry_text, root

    # 设置窗口
    root = setup_window()

    # 设置组件
    label_img1, label_img2, entry_text = setup_widgets(root)

    # 运行Tkinter事件循环
    root.mainloop()

if __name__ == '__main__':
    main()
