from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import sys
args  = sys.argv

# 画像を読み込む
image_dir: str = args[1]
image = Image.open(image_dir)
image_np = np.array(image)

# Tkinterウィンドウを作成
root = tk.Tk()

# 画像をウィンドウに表示
photo = ImageTk.PhotoImage(image=image)
canvas = tk.Canvas(root, width=image.width, height=image.height)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=photo)

# 領域選択用の矩形を描画
rect = None
start_x, start_y = None, None

def start_rect(event) -> None:
    global rect, start_x, start_y
    start_x, start_y = event.x, event.y
    rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

def update_rect(event):
    global rect
    canvas.coords(rect, start_x, start_y, event.x, event.y)

def end_rect(event) -> None:
    global rect
    x1, y1, x2, y2 = [int(x) for x in canvas.coords(rect)]  # ここを変更
    canvas.delete(rect)

    # 選択領域のRGB平均値を計算
    region = image_np[y1:y2, x1:x2]
    mean_r = np.mean(region[:, :, 0])
    mean_g = np.mean(region[:, :, 1])
    mean_b = np.mean(region[:, :, 2])
    print(f"選択領域の最初のピクセル位置: ({start_x}, {start_y})")
    print(f"選択領域の最後のピクセル位置: ({event.x}, {event.y})")
    print(f"選択領域のRGB平均値: ({mean_r:.2f}, {mean_g:.2f}, {mean_b:.2f})")

# イベントバインド
canvas.bind("<Button-1>", start_rect)
canvas.bind("<B1-Motion>", update_rect)
canvas.bind("<ButtonRelease-1>", end_rect)

root.mainloop()