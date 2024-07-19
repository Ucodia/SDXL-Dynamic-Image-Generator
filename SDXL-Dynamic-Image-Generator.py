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
        self.window.geometry('1200x800')  # Adjusted for side-by-side layout

        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left panel for controls
        self.controls_frame = ttk.Frame(self.main_frame, width=300)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.controls_frame.pack_propagate(False)  # Prevent frame from shrinking

        # Create right panel for canvas
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.output_canvas = tk.Canvas(self.canvas_frame, width=512, height=512, bg='lightgray')
        self.output_canvas.pack(fill=tk.BOTH, expand=True)

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

        self.text_input = tk.Text(self.controls_frame, width=30, height=5, font=font_size, wrap=tk.WORD)
        self.text_input.insert(tk.END, default_prompt)
        self.text_input.pack(fill=tk.X, pady=(0, 10))

        # Negative prompt input
        self.negative_prompt_label = ttk.Label(self.controls_frame, text="Negative Prompt:", font=font_size)
        self.negative_prompt_label.pack(anchor='w')

        self.negative_prompt_input = tk.Text(self.controls_frame, width=30, height=3, font=font_size, wrap=tk.WORD)
        self.negative_prompt_input.insert(tk.END, "nsfw, naked")
        self.negative_prompt_input.pack(fill=tk.X, pady=(0, 10))

        # Sliders
        sliders = [
            ("Strength", 0.0, 1.0, 0.1, 1.0),
            ("Guidance Scale", 0.0, 1.0, 0.1, 1.0),
            ("Num Inference Steps", 1, 50, 1, 2),
            ("Seed", 0, 10000, 1, 1),
            ("Width", 256, 1024, 64, 512),
            ("Height", 256, 1024, 64, 512)
        ]

        for label, from_, to, resolution, default in sliders:
            slider_frame = ttk.Frame(self.controls_frame)
            slider_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(slider_frame, text=label, font=font_size).pack(anchor='w')
            
            slider = tk.Scale(slider_frame, from_=from_, to=to, resolution=resolution,
                              orient=tk.HORIZONTAL, length=280)
            slider.set(default)
            slider.pack(fill=tk.X)
            
            setattr(self, f"{label.lower().replace(' ', '_')}_slider", slider)

        self.btn_toggle_record = ttk.Button(self.controls_frame, text="Toggle Generation", command=self.toggle_recording, style='W.TButton')
        self.btn_toggle_record.pack(pady=10, fill=tk.X)

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
        negative_prompt = self.negative_prompt_input.get("1.0", tk.END).strip()
        seed = int(self.seed_slider.get())
        width = int(self.width_slider.get())
        height = int(self.height_slider.get())

        if prompt:
            torch.manual_seed(seed)

            if self.previous_frame is None:
                init_image = Image.new('RGB', (width, height), color='white')
                transformed_image = self.pipe(prompt=prompt,
                                              negative_prompt=negative_prompt,
                                              image=init_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=int(self.num_inference_steps_slider.get())).images[0]
                self.previous_frame = transformed_image
            else:
                perturbed_image = self.apply_random_perturbations(self.previous_frame)
                perturbed_image = perturbed_image.resize((width, height), Image.LANCZOS)
                transformed_image = self.pipe(prompt=prompt,
                                              negative_prompt=negative_prompt,
                                              image=perturbed_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=int(self.num_inference_steps_slider.get())).images[0]
                blended_image = self.blend_images(self.previous_frame.resize((width, height), Image.LANCZOS), transformed_image, alpha=0.1)
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
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))
        return image

    def blend_images(self, prev_img, curr_img, alpha=0.1):
        prev_array = np.array(prev_img)
        curr_array = np.array(curr_img)
        blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)
        return Image.fromarray(blended_array)

    def display_transformed_image(self, transformed_image):
        photo = ImageTk.PhotoImage(transformed_image)
        self.output_canvas.delete("all")
        self.output_canvas.config(width=transformed_image.width, height=transformed_image.height)
        self.output_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.output_canvas.image = photo

    def on_closing(self):
        self.recording = False
        if self.generation_thread:
            self.generation_thread.join()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = ImageGeneratorApp(root, "SDXL Live")
    root.mainloop()

if __name__ == '__main__':
    main()