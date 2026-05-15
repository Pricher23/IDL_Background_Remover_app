import cv2
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import os
import shutil
from tkinter import filedialog, simpledialog
from segmentation import get_mask, get_mask_custom, load_custom_model, apply_background
from background import load_backgrounds, solid_color_background
from llm_controller import ask_llm
from segmentation import get_mask, get_mask_custom, load_custom_model, apply_background, apply_blur_background
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BACKGROUNDS_FOLDER = "backgrounds"
# using lists so the thread can see changes without global declarations
running = [True]
use_custom_model = [False]
current_bg_name = [None]
bg_images = {}
bg_buttons = {}

# load once at startup so we don't do it every frame
load_custom_model("model_weights.pth")
mp_selfie = mp.solutions.selfie_segmentation

app = ctk.CTk()
app.title("Virtual Background")
app.geometry("1100x680")
app.resizable(True, True)
app.grid_columnconfigure(0, weight=3)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

left_frame = ctk.CTkFrame(app, corner_radius=12, fg_color="transparent")
left_frame.grid(row=0, column=0, sticky="nsew", padx=(16,8), pady=16)
left_frame.grid_rowconfigure(1, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

title_row = ctk.CTkFrame(left_frame, fg_color="transparent")
title_row.grid(row=0, column=0, sticky="ew", pady=(0,10))
title_row.grid_columnconfigure(1, weight=1)

ctk.CTkLabel(title_row, text="Virtual Background",
             font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold")).grid(row=0, column=0, sticky="w")

theme_state = ["light"]
def toggle_theme():
    if theme_state[0] == "light":
        ctk.set_appearance_mode("dark")
        theme_state[0] = "dark"
        theme_btn.configure(text="☀  Light mode")
    else:
        ctk.set_appearance_mode("light")
        theme_state[0] = "light"
        theme_btn.configure(text="🌙  Dark mode")

theme_btn = ctk.CTkButton(title_row, text="🌙  Dark mode", width=120, height=30,
                           font=ctk.CTkFont(size=12), command=toggle_theme)
theme_btn.grid(row=0, column=2, sticky="e", padx=(8,0))

video_label = ctk.CTkLabel(left_frame, text="Starting camera…", width=640, height=480,
                            corner_radius=10)
video_label.grid(row=1, column=0, sticky="nsew")
chat_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
chat_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
chat_frame.grid_columnconfigure(0, weight=1)

chat_log = ctk.CTkTextbox(chat_frame, height=80, state="disabled",
                           font=ctk.CTkFont(size=12), corner_radius=8)
chat_log.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

chat_input = ctk.CTkEntry(chat_frame, placeholder_text="Type a command…",
                           font=ctk.CTkFont(size=13), height=36, corner_radius=8)
chat_input.grid(row=1, column=0, sticky="ew", padx=(0, 8))

def send_command():
    user_text = chat_input.get().strip()
    if not user_text:
        return
    chat_input.delete(0, "end")

    # show in log
    chat_log.configure(state="normal")
    chat_log.insert("end", f"You: {user_text}\n")
    chat_log.configure(state="disabled")

    # run in thread so GUI doesn't freeze
    import threading
    def run():
        bg_names = list(bg_images.keys())
        result = ask_llm(user_text, bg_names)
        action = result.get("action")
        value = result.get("value", "")

        if action == "set_color":
            bgr = result.get("value", [0, 200, 0])
            if isinstance(bgr, list) and len(bgr) == 3:
                bgr = tuple(bgr)
            else:
                bgr = (0, 200, 0)  # fallback to green if LLM messes up
            bg_images["__solid__"] = solid_color_background(480, 640, color=bgr)
            current_bg_name[0] = "__solid__"
        elif action == "set_image":
            match = next((n for n in bg_images if value.lower() in n.lower()), None)
            if match:
                switch_bg(match)
        elif action == "reset":
            if bg_images:
                switch_bg(list(bg_images.keys())[0])

        chat_log.configure(state="normal")
        chat_log.insert("end", f"App: {action} → {value}\n")
        chat_log.configure(state="disabled")

    threading.Thread(target=run, daemon=True).start()

chat_input.bind("<Return>", lambda e: send_command())

send_btn = ctk.CTkButton(chat_frame, text="Send", width=80, height=36,
                          font=ctk.CTkFont(size=13), command=send_command)
send_btn.grid(row=1, column=1, sticky="e")
model_bar = ctk.CTkFrame(left_frame, fg_color="transparent")
model_bar.grid(row=2, column=0, sticky="ew", pady=(10,0))
model_bar.grid_columnconfigure(1, weight=1)

model_badge = ctk.CTkLabel(model_bar, text="● MediaPipe  (baseline)",
                            font=ctk.CTkFont(size=12), text_color="#0ea5e9")
model_badge.grid(row=0, column=0, sticky="w")

def toggle_model():
    use_custom_model[0] = not use_custom_model[0]
    if use_custom_model[0]:
        model_badge.configure(text="● MobileNetV3  (fine-tuned)", text_color="#22c55e")
        model_toggle_btn.configure(text="Switch to MediaPipe")
    else:
        model_badge.configure(text="● MediaPipe  (baseline)", text_color="#0ea5e9")
        model_toggle_btn.configure(text="Switch to MobileNetV3")

model_toggle_btn = ctk.CTkButton(model_bar, text="Switch to MobileNetV3", width=180,
                                  height=30, font=ctk.CTkFont(size=12), command=toggle_model)
model_toggle_btn.grid(row=0, column=2, sticky="e")

use_blur = [False]

def toggle_blur():
    use_blur[0] = not use_blur[0]
    blur_btn.configure(text="Blur: ON" if use_blur[0] else "Blur: OFF",
                       fg_color="#8b5cf6" if use_blur[0] else None)

blur_btn = ctk.CTkButton(model_bar, text="Blur: OFF", width=100,
                          height=30, font=ctk.CTkFont(size=12), command=toggle_blur)
blur_btn.grid(row=0, column=3, sticky="e", padx=(8,0))

right_frame = ctk.CTkFrame(app, corner_radius=12)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(8,16), pady=16)
right_frame.grid_rowconfigure(1, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

sidebar_header = ctk.CTkFrame(right_frame, fg_color="transparent")
sidebar_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12,6))
sidebar_header.grid_columnconfigure(0, weight=1)

ctk.CTkLabel(sidebar_header, text="Backgrounds",
             font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")

add_btn = ctk.CTkButton(sidebar_header, text="+ Add", width=60, height=28,
                         font=ctk.CTkFont(size=12))
add_btn.grid(row=0, column=1, sticky="e")

scroll = ctk.CTkScrollableFrame(right_frame, corner_radius=8)
scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))
scroll.grid_columnconfigure(0, weight=1)

def load_all_backgrounds():
    global bg_images
    bg_images = load_backgrounds(BACKGROUNDS_FOLDER)
    for w in scroll.winfo_children():
        w.destroy()
    bg_buttons.clear()
    for name in list(bg_images.keys()):
        _add_bg_card(name, prepend=False)
    if bg_images:
        first = list(bg_images.keys())[0]
        current_bg_name[0] = first
        _mark_selected(first)

def _add_bg_card(name, prepend=True):
    card = ctk.CTkFrame(scroll, corner_radius=8, height=56)
    card.grid_columnconfigure(1, weight=1)
    img_cv = bg_images.get(name)
    if img_cv is not None:
        thumb = cv2.resize(img_cv, (64, 40))
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        pil_thumb = Image.fromarray(thumb_rgb)
        ctk_thumb = ctk.CTkImage(light_image=pil_thumb, dark_image=pil_thumb, size=(64, 40))
        thumb_lbl = ctk.CTkLabel(card, image=ctk_thumb, text="")
        thumb_lbl.grid(row=0, column=0, padx=(8,6), pady=8)
        thumb_lbl.bind("<Button-1>", lambda e, n=name: switch_bg(n))
    name_lbl = ctk.CTkLabel(card, text=name.capitalize(),
                             font=ctk.CTkFont(size=13), anchor="w")
    name_lbl.grid(row=0, column=1, sticky="ew", padx=(0,4))
    name_lbl.bind("<Button-1>", lambda e, n=name: switch_bg(n))
    card.bind("<Button-1>", lambda e, n=name: switch_bg(n))
    del_btn = ctk.CTkButton(card, text="✕", width=28, height=28, fg_color="transparent",
                             text_color="gray", hover_color="#fee2e2",
                             font=ctk.CTkFont(size=12),
                             command=lambda n=name: delete_bg(n))
    del_btn.grid(row=0, column=2, padx=(0,6), pady=8)
    if prepend:
        # shift everything down so the new card goes on top
        for w in scroll.winfo_children():
            info = w.grid_info()
            if info:
                w.grid(row=int(info["row"]) + 1, column=0, sticky="ew", padx=4, pady=3)
        card.grid(row=0, column=0, sticky="ew", padx=4, pady=3)
    else:
        row = len(bg_buttons)
        card.grid(row=row, column=0, sticky="ew", padx=4, pady=3)
    bg_buttons[name] = (card, name_lbl, del_btn)

def _mark_selected(name):
    for n, (card, lbl, _) in bg_buttons.items():
        if n == name:
            card.configure(border_width=2, border_color="#3b82f6")
        else:
            card.configure(border_width=0)

def switch_bg(name):
    current_bg_name[0] = name
    _mark_selected(name)

def delete_bg(name):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(BACKGROUNDS_FOLDER, name + ext)
        if os.path.exists(path):
            os.remove(path)
            break
    bg_images.pop(name, None)
    if name in bg_buttons:
        bg_buttons[name][0].destroy()
        del bg_buttons[name]
    if bg_images:
        first = list(bg_images.keys())[0]
        current_bg_name[0] = first
        _mark_selected(first)
    else:
        current_bg_name[0] = None

def add_background():
    path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return
    name = simpledialog.askstring("Background name",
                                   "Enter a name for this background:",
                                   parent=app)
    if not name:
        return
    name = name.strip().lower().replace(" ", "_")
    ext = os.path.splitext(path)[1].lower()
    dest = os.path.join(BACKGROUNDS_FOLDER, name + ext)
    shutil.copy2(path, dest)
    img = cv2.imread(dest)
    if img is not None:
        bg_images[name] = img
        if name in bg_buttons:
            bg_buttons[name][0].destroy()
            del bg_buttons[name]
        _add_bg_card(name, prepend=True)
        switch_bg(name)

add_btn.configure(command=add_background)
load_all_backgrounds()

cap = cv2.VideoCapture(0)

def video_loop():
    # model_selection=1 is the landscape/higher quality mode
    with mp_selfie.SelfieSegmentation(model_selection=1) as model:
        while running[0]:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # mirror like a webcam should look
            if not ret:
                continue
            h, w = frame.shape[:2]
            name = current_bg_name[0]
            # fall back to plain green if no background is selected
            bg = bg_images.get(name, solid_color_background(h, w)) if name else solid_color_background(h, w)
            if use_custom_model[0]:
                mask = get_mask_custom(frame)
            else:
                mask = get_mask(frame, model)
            if use_blur[0]:
                output = apply_blur_background(frame, mask)
            else:
                output = apply_background(frame, mask, bg)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(output_rgb).resize((640, 460))
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 460))
            video_label.configure(image=ctk_img, text="")
            # need to keep a reference or tkinter garbage-collects the image
            video_label.image = ctk_img

def on_close():
    running[0] = False
    cap.release()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
threading.Thread(target=video_loop, daemon=True).start()
app.mainloop()
