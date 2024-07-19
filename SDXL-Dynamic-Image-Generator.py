import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance
import torch
from diffusers import AutoPipelineForImage2Image
import numpy as np
import random
import threading
import queue

class ImageGeneratorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('600x800')

        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.output_canvas = tk.Canvas(self.main_frame, width=512, height=512, bg='lightgray')
        self.output_canvas.pack(pady=10)

        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

        self.setup_ui()

        self.recording = False
        self.previous_frame = None
        self.frame_count = 0

        self.image_queue = queue.Queue(maxsize=1)
        self.generation_thread = None

        self.load_model()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        font_size = ('Helvetica', 16)
        style = ttk.Style()
        style.configure('W.TButton', font=('Helvetica', 16))

        # Default text prompt
        default_prompt = "A photograph of a water drop"

        # Text input prompt
        self.text_input_label = ttk.Label(self.controls_frame, text="Prompt:", font=font_size)
        self.text_input_label.pack(anchor='w')

        self.text_input = tk.Text(self.controls_frame, width=60, height=2, font=font_size, wrap=tk.WORD)
        self.text_input.insert(tk.END, default_prompt)
        self.text_input.pack(fill=tk.X, pady=5)

        # Frame for sliders
        sliders_frame = ttk.Frame(self.controls_frame)
        sliders_frame.pack(fill=tk.X, pady=10)

        # Adjusted slider ranges and defaults
        self.strength_slider = tk.Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Strength", length=150)
        self.strength_slider.set(1.0)  # Set default
        self.strength_slider.pack(side=tk.LEFT, padx=5)

        self.guidance_scale_slider = tk.Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Guidance Scale", length=150)
        self.guidance_scale_slider.set(1.0)  # Set default
        self.guidance_scale_slider.pack(side=tk.LEFT, padx=5)

        self.num_steps_slider = tk.Scale(sliders_frame, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL, label="Num Inference Steps", length=150)
        self.num_steps_slider.set(2)  # Set default
        self.num_steps_slider.pack(side=tk.LEFT, padx=5)

        self.seed_slider = tk.Scale(sliders_frame, from_=0, to=10000, resolution=1, orient=tk.HORIZONTAL, label="Seed", length=150)
        self.seed_slider.set(1)  # Set default to 1
        self.seed_slider.pack(side=tk.LEFT, padx=5)

        self.btn_toggle_record = ttk.Button(self.controls_frame, text="Toggle Generation", command=self.toggle_recording, width=20, style='W.TButton')
        self.btn_toggle_record.pack(pady=10)

    def load_model(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to(device)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            print("Generation started...")
            self.generation_thread = threading.Thread(target=self.generate_images)
            self.generation_thread.start()
            self.check_queue()
        else:
            print("Generation stopped.")

    def generate_images(self):
        while self.recording:
            self.process_frame()

    def process_frame(self):
        prompt = self.text_input.get("1.0", tk.END).strip()
        seed = self.seed_slider.get()

        if prompt:
            torch.manual_seed(seed)

            if self.previous_frame is None:
                init_image = Image.new('RGB', (512, 512), color='white')
                transformed_image = self.pipe(prompt=prompt,
                                              image=init_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=self.num_steps_slider.get()).images[0]
                self.previous_frame = transformed_image
            else:
                perturbed_image = self.apply_random_perturbations(self.previous_frame)
                transformed_image = self.pipe(prompt=prompt,
                                              image=perturbed_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=self.num_steps_slider.get()).images[0]
                blended_image = self.blend_images(self.previous_frame, transformed_image, alpha=0.1)
                self.previous_frame = blended_image

                try:
                    self.image_queue.put(blended_image, block=False)
                except queue.Full:
                    pass  # Skip this frame if the queue is full

            self.frame_count += 1
            if self.frame_count % 20 == 0:
                self.window.after(0, lambda: self.seed_slider.set(seed + 1))

    def check_queue(self):
        try:
            image = self.image_queue.get(block=False)
            self.display_transformed_image(image)
        except queue.Empty:
            pass
        
        if self.recording:
            self.window.after(10, self.check_queue)

    def apply_random_perturbations(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))  # Adjust brightness slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))  # Adjust contrast slightly
        return image

    def blend_images(self, prev_img, curr_img, alpha=0.1):
        prev_array = np.array(prev_img)
        curr_array = np.array(curr_img)
        blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)
        return Image.fromarray(blended_array)

    def display_transformed_image(self, transformed_image):
        photo = ImageTk.PhotoImage(transformed_image)
        self.output_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.output_canvas.image = photo

    def on_closing(self):
        self.recording = False
        if self.generation_thread:
            self.generation_thread.join()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = ImageGeneratorApp(root, "Image Generator App")
    root.mainloop()

if __name__ == '__main__':
    main()