import tkinter as tk
import random
import time

class BCIVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI Motor Imagery Classification")
        self.root.geometry("600x400")
        self.root.configure(bg='black')
        
        # Title label
        self.title_label = tk.Label(
            root, 
            text="Brain-Computer Interface Visualization",
            font=("Arial", 24, "bold"),
            bg='black', fg='white'
        )
        self.title_label.pack(pady=20)
        
        # Current state display
        self.state_frame = tk.Frame(root, bg='black')
        self.state_frame.pack(pady=20)
        
        self.state_label = tk.Label(
            self.state_frame,
            text="Current State:",
            font=("Arial", 18),
            bg='black', fg='white'
        )
        self.state_label.pack(side=tk.LEFT, padx=10)
        
        self.current_state = tk.StringVar()
        self.current_state.set("INITIALIZING...")
        
        self.state_value = tk.Label(
            self.state_frame,
            textvariable=self.current_state,
            font=("Arial", 24, "bold"),
            bg='black', fg='green'
        )
        self.state_value.pack(side=tk.LEFT, padx=10)
        
        # State boxes
        self.boxes_frame = tk.Frame(root, bg='black')
        self.boxes_frame.pack(pady=40)
        
        # Rest box
        self.rest_frame = tk.Frame(
            self.boxes_frame, 
            width=100, 
            height=100,
            bg='gray'
        )
        self.rest_frame.pack(side=tk.LEFT, padx=20)
        self.rest_label = tk.Label(
            self.rest_frame,
            text="REST",
            font=("Arial", 16, "bold"),
            bg='gray', fg='white',
            width=8, height=4
        )
        self.rest_label.pack()
        
        # Left box
        self.left_frame = tk.Frame(
            self.boxes_frame, 
            width=100, 
            height=100,
            bg='gray'
        )
        self.left_frame.pack(side=tk.LEFT, padx=20)
        self.left_label = tk.Label(
            self.left_frame,
            text="LEFT",
            font=("Arial", 16, "bold"),
            bg='gray', fg='white',
            width=8, height=4
        )
        self.left_label.pack()
        
        # Right box
        self.right_frame = tk.Frame(
            self.boxes_frame, 
            width=100, 
            height=100,
            bg='gray'
        )
        self.right_frame.pack(side=tk.LEFT, padx=20)
        self.right_label = tk.Label(
            self.right_frame,
            text="RIGHT",
            font=("Arial", 16, "bold"),
            bg='gray', fg='white',
            width=8, height=4
        )
        self.right_label.pack()
        
        # Status area
        self.status_label = tk.Label(
            root,
            text="Simulation Mode",
            font=("Arial", 10),
            bg='black', fg='yellow'
        )
        self.status_label.pack(pady=20)
        
        # Start cycling states
        self.class_names = ['rest', 'left', 'right']
        self.cycle_states()
    
    def update_state(self, state):
        """Update the displayed state"""
        self.current_state.set(state.upper())
        
        # Reset all boxes to gray
        self.rest_frame.config(bg='gray')
        self.rest_label.config(bg='gray')
        self.left_frame.config(bg='gray')
        self.left_label.config(bg='gray')
        self.right_frame.config(bg='gray')
        self.right_label.config(bg='gray')
        
        # Highlight the active state
        if state == 'rest':
            self.rest_frame.config(bg='green')
            self.rest_label.config(bg='green')
            self.state_value.config(fg='green')
        elif state == 'left':
            self.left_frame.config(bg='blue')
            self.left_label.config(bg='blue')
            self.state_value.config(fg='blue')
        elif state == 'right':
            self.right_frame.config(bg='red')
            self.right_label.config(bg='red')
            self.state_value.config(fg='red')
    
    def cycle_states(self):
        """Cycle through different states"""
        state = random.choice(self.class_names)
        self.update_state(state)
        # Schedule the next update
        self.root.after(1500, self.cycle_states)  # Change state every 1.5 seconds

if __name__ == "__main__":
    root = tk.Tk()
    app = BCIVisualization(root)
    root.mainloop()