import tkinter as tk
import numpy as np
import joblib
import time
import threading
import os
import random
import math

# Try to load the model
try:
    model = joblib.load('high_acc_gb_model.pkl')  # You can change this to your preferred model
    print("Model loaded successfully!")
    have_model = True
except Exception as e:
    print(f"Could not load model: {e}")
    print("Will run in simulation mode")
    have_model = False

# Class names
class_names = ['rest', 'left', 'right']

class ProstheticArmSimulation:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI Prosthetic Arm Simulation")
        self.root.geometry("1000x700")
        self.root.configure(bg='#202020')
        
        # Main frame
        self.main_frame = tk.Frame(root, bg='#202020')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="Brain-Computer Interface Prosthetic Control",
            font=("Arial", 24, "bold"),
            bg='#202020', fg='white'
        )
        self.title_label.pack(pady=20)
        
        # Status display
        self.status_frame = tk.Frame(self.main_frame, bg='#202020')
        self.status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Current Mental Command:",
            font=("Arial", 14),
            bg='#202020', fg='white'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.current_state = tk.StringVar()
        self.current_state.set("Initializing...")
        
        self.state_value = tk.Label(
            self.status_frame,
            textvariable=self.current_state,
            font=("Arial", 18, "bold"),
            bg='#202020', fg='#00FF00'
        )
        self.state_value.pack(side=tk.LEFT, padx=10)
        
        # Canvas for arm visualization
        self.canvas_width = 800
        self.canvas_height = 500
        self.canvas = tk.Canvas(
            self.main_frame, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg='#303030',
            highlightthickness=1,
            highlightbackground='#505050'
        )
        self.canvas.pack(pady=20)
        
        # Initial arm parameters
        self.upper_arm_length = 150
        self.forearm_length = 120
        self.hand_length = 60
        self.hand_width = 40
        
        # Arm position and angles
        self.shoulder_x = 200
        self.shoulder_y = 200
        self.shoulder_angle = 0
        self.elbow_angle = 0
        self.wrist_angle = 0
        self.grip_state = 'closed'  # 'open' or 'closed'
        
        # Draw the initial arm
        self.draw_arm()
        
        # Information panel
        self.info_frame = tk.Frame(self.main_frame, bg='#202020')
        self.info_frame.pack(fill=tk.X, pady=10)
        
        # Legend
        self.legend_frame = tk.Frame(self.info_frame, bg='#202020')
        self.legend_frame.pack(side=tk.LEFT, padx=20)
        
        # Rest command
        self.rest_frame = tk.Frame(self.legend_frame, bg='#202020')
        self.rest_frame.pack(anchor=tk.W, pady=5)
        
        self.rest_color = tk.Frame(self.rest_frame, width=20, height=20, bg='#00FF00')
        self.rest_color.pack(side=tk.LEFT, padx=5)
        
        self.rest_text = tk.Label(
            self.rest_frame,
            text="REST: Hold position",
            font=("Arial", 12),
            bg='#202020', fg='white'
        )
        self.rest_text.pack(side=tk.LEFT)
        
        # Left command
        self.left_frame = tk.Frame(self.legend_frame, bg='#202020')
        self.left_frame.pack(anchor=tk.W, pady=5)
        
        self.left_color = tk.Frame(self.left_frame, width=20, height=20, bg='#2196F3')
        self.left_color.pack(side=tk.LEFT, padx=5)
        
        self.left_text = tk.Label(
            self.left_frame,
            text="LEFT: Close/Open hand",
            font=("Arial", 12),
            bg='#202020', fg='white'
        )
        self.left_text.pack(side=tk.LEFT)
        
        # Right command
        self.right_frame = tk.Frame(self.legend_frame, bg='#202020')
        self.right_frame.pack(anchor=tk.W, pady=5)
        
        self.right_color = tk.Frame(self.right_frame, width=20, height=20, bg='#F44336')
        self.right_color.pack(side=tk.LEFT, padx=5)
        
        self.right_text = tk.Label(
            self.right_frame,
            text="RIGHT: Rotate wrist",
            font=("Arial", 12),
            bg='#202020', fg='white'
        )
        self.right_text.pack(side=tk.LEFT)
        
        # Model info
        self.model_label = tk.Label(
            self.info_frame,
            text="Simulation Mode" if not have_model else "Using Trained Model",
            font=("Arial", 10),
            bg='#202020', fg='yellow'
        )
        self.model_label.pack(side=tk.RIGHT, padx=20)
        
        # Start the classification thread
        self.stop_flag = False
        self.classification_thread = threading.Thread(target=self.simulate_classification)
        self.classification_thread.daemon = True
        self.classification_thread.start()
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def draw_arm(self):
        """Draw the prosthetic arm on the canvas"""
        self.canvas.delete("all")
        
        # Calculate joint positions
        elbow_x = self.shoulder_x + self.upper_arm_length * math.cos(math.radians(self.shoulder_angle))
        elbow_y = self.shoulder_y + self.upper_arm_length * math.sin(math.radians(self.shoulder_angle))
        
        wrist_x = elbow_x + self.forearm_length * math.cos(math.radians(self.shoulder_angle + self.elbow_angle))
        wrist_y = elbow_y + self.forearm_length * math.sin(math.radians(self.shoulder_angle + self.elbow_angle))
        
        # Calculate hand points
        hand_angle = self.shoulder_angle + self.elbow_angle + self.wrist_angle
        hand_end_x = wrist_x + self.hand_length * math.cos(math.radians(hand_angle))
        hand_end_y = wrist_y + self.hand_length * math.sin(math.radians(hand_angle))
        
        # Calculate hand width points
        hand_width_angle = hand_angle + 90
        half_width = self.hand_width / 2
        
        if self.grip_state == 'open':
            # Open hand (wider)
            width_factor = 1.5
        else:
            # Closed hand (narrower)
            width_factor = 0.5
            
        hand_top_x = wrist_x + half_width * width_factor * math.cos(math.radians(hand_width_angle))
        hand_top_y = wrist_y + half_width * width_factor * math.sin(math.radians(hand_width_angle))
        
        hand_bottom_x = wrist_x - half_width * width_factor * math.cos(math.radians(hand_width_angle))
        hand_bottom_y = wrist_y - half_width * width_factor * math.sin(math.radians(hand_width_angle))
        
        # Draw shoulder joint
        self.canvas.create_oval(
            self.shoulder_x - 10, self.shoulder_y - 10,
            self.shoulder_x + 10, self.shoulder_y + 10,
            fill='#888888', outline='#666666', width=2
        )
        
        # Draw upper arm
        self.canvas.create_line(
            self.shoulder_x, self.shoulder_y, elbow_x, elbow_y,
            fill='#AAAAAA', width=15, capstyle=tk.ROUND
        )
        
        # Draw elbow joint
        self.canvas.create_oval(
            elbow_x - 8, elbow_y - 8,
            elbow_x + 8, elbow_y + 8,
            fill='#888888', outline='#666666', width=2
        )
        
        # Draw forearm
        self.canvas.create_line(
            elbow_x, elbow_y, wrist_x, wrist_y,
            fill='#AAAAAA', width=12, capstyle=tk.ROUND
        )
        
        # Draw wrist joint
        self.canvas.create_oval(
            wrist_x - 6, wrist_y - 6,
            wrist_x + 6, wrist_y + 6,
            fill='#888888', outline='#666666', width=2
        )
        
        # Draw hand
        if self.grip_state == 'open':
            # Draw open hand (like a palm)
            self.canvas.create_polygon(
                wrist_x, wrist_y,
                hand_top_x, hand_top_y,
                hand_end_x, hand_end_y,
                hand_bottom_x, hand_bottom_y,
                fill='#DDDDDD', outline='#999999', width=1
            )
            
            # Draw fingers
            finger_length = self.hand_length * 0.6
            finger_width = self.hand_width * 0.15
            finger_spacing = self.hand_width * 0.25
            
            for i in range(3):  # Draw 3 fingers
                finger_base_offset = (i - 1) * finger_spacing
                finger_base_x = hand_end_x - finger_length * 0.2 * math.cos(math.radians(hand_angle))
                finger_base_y = hand_end_y - finger_length * 0.2 * math.sin(math.radians(hand_angle))
                
                # Offset the finger base
                finger_base_x += finger_base_offset * math.cos(math.radians(hand_width_angle))
                finger_base_y += finger_base_offset * math.sin(math.radians(hand_width_angle))
                
                finger_end_x = finger_base_x + finger_length * math.cos(math.radians(hand_angle))
                finger_end_y = finger_base_y + finger_length * math.sin(math.radians(hand_angle))
                
                self.canvas.create_line(
                    finger_base_x, finger_base_y, finger_end_x, finger_end_y,
                    fill='#CCCCCC', width=finger_width, capstyle=tk.ROUND
                )
        else:
            # Draw closed hand (like a fist)
            self.canvas.create_polygon(
                wrist_x, wrist_y,
                hand_top_x, hand_top_y,
                hand_end_x, hand_end_y,
                hand_bottom_x, hand_bottom_y,
                fill='#DDDDDD', outline='#999999', width=1
            )
        
        # Draw base plate
        plate_width = 100
        plate_height = 30
        self.canvas.create_rectangle(
            self.shoulder_x - plate_width/2, self.shoulder_y - 30,
            self.shoulder_x + plate_width/2, self.shoulder_y,
            fill='#444444', outline='#333333', width=2
        )
        
        # Draw command indicators
        if self.current_state.get() == "REST":
            indicator_color = '#00FF00'  # Green
        elif self.current_state.get() == "LEFT":
            indicator_color = '#2196F3'  # Blue
        elif self.current_state.get() == "RIGHT":
            indicator_color = '#F44336'  # Red
        else:
            indicator_color = '#AAAAAA'  # Gray
            
        self.canvas.create_rectangle(
            self.shoulder_x - 40, self.shoulder_y - 20,
            self.shoulder_x + 40, self.shoulder_y - 10,
            fill=indicator_color, outline='#FFFFFF', width=1
        )
    
    def update_arm(self, command):
        """Update the arm based on brain command"""
        if command == 'rest':
            # Just hold current position
            pass
        elif command == 'left':
            # Toggle hand open/closed
            if self.grip_state == 'closed':
                self.grip_state = 'open'
            else:
                self.grip_state = 'closed'
        elif command == 'right':
            # Rotate wrist
            self.wrist_angle = (self.wrist_angle + 30) % 360
        
        # Redraw the arm
        self.draw_arm()
    
    def update_state(self, new_state):
        """Update the displayed state and arm"""
        self.current_state.set(new_state.upper())
        
        # Update text color based on state
        if new_state == 'rest':
            self.state_value.config(fg='#00FF00')  # Green
        elif new_state == 'left':
            self.state_value.config(fg='#2196F3')  # Blue
        elif new_state == 'right':
            self.state_value.config(fg='#F44336')  # Red
        
        # Update the arm
        self.update_arm(new_state)
    
    def simulate_classification(self):
        """Simulate BCI classification at fixed intervals"""
        while not self.stop_flag:
            # In a real application, you would get data from an EEG device,
            # process it, extract features, and use your model
            
            if have_model:
                # This should match your feature extraction
                feature_vector = np.random.random((1, 84))  # Adjust based on your feature dimensions
                
                try:
                    prediction = model.predict(feature_vector)
                    predicted_class = class_names[prediction[0] % len(class_names)]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_class = random.choice(class_names)
            else:
                # Simulation mode
                predicted_class = random.choice(class_names)
            
            # Update the GUI (in the main thread)
            self.root.after(0, self.update_state, predicted_class)
            
            # Wait before next prediction
            time.sleep(2)
    
    def on_closing(self):
        """Clean up before closing"""
        self.stop_flag = True
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProstheticArmSimulation(root)
    root.mainloop()