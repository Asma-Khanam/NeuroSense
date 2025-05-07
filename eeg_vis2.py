import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import random
import time

class EEGStateVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Motor Imagery Pattern Visualization")
        self.root.geometry("1200x800")
        self.root.configure(bg='#202020')
        
        # Sampling parameters
        self.sampling_rate = 250  # Hz
        self.signal_length = 3    # seconds
        self.buffer_size = self.sampling_rate * self.signal_length
        self.time_axis = np.linspace(0, self.signal_length, self.buffer_size)
        
        # Key EEG channels for motor imagery
        self.channels = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']
        
        # Create frame for buttons
        self.button_frame = tk.Frame(self.root, bg='#202020')
        self.button_frame.pack(pady=10)
        
        # Create buttons for each state
        self.rest_button = tk.Button(
            self.button_frame, 
            text="REST State",
            command=lambda: self.open_state_window('rest'),
            font=("Arial", 14),
            bg='#00FF00',
            fg='black',
            width=15,
            height=2
        )
        self.rest_button.pack(side=tk.LEFT, padx=20)
        
        self.left_button = tk.Button(
            self.button_frame, 
            text="LEFT Hand MI",
            command=lambda: self.open_state_window('left'),
            font=("Arial", 14),
            bg='#2196F3',
            fg='white',
            width=15,
            height=2
        )
        self.left_button.pack(side=tk.LEFT, padx=20)
        
        self.right_button = tk.Button(
            self.button_frame, 
            text="RIGHT Hand MI",
            command=lambda: self.open_state_window('right'),
            font=("Arial", 14),
            bg='#F44336',
            fg='white',
            width=15,
            height=2
        )
        self.right_button.pack(side=tk.LEFT, padx=20)
        
        # Create a frame for the main visualization
        self.main_frame = tk.Frame(self.root, bg='#202020')
        self.main_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="EEG Motor Imagery Patterns - All States Comparison",
            font=("Arial", 18, "bold"),
            bg='#202020', fg='white'
        )
        self.title_label.pack(pady=10)
        
        # Create the main figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#303030')
        
        # Create subplots for each channel
        self.axes = []
        for i in range(3):
            ax = self.fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('#303030')
            ax.set_title(self.channels[i], color='white', fontsize=12)
            ax.set_ylabel('Amplitude (μV)', color='white')
            ax.set_xlim(0, self.signal_length)
            ax.set_ylim(-2, 2)
            if i == 2:  # Only add x-label for the bottom plot
                ax.set_xlabel('Time (s)', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Initialize lines for each state
            rest_line, = ax.plot([], [], color='#00FF00', linewidth=2, label='Rest')
            left_line, = ax.plot([], [], color='#2196F3', linewidth=2, label='Left MI')
            right_line, = ax.plot([], [], color='#F44336', linewidth=2, label='Right MI')
            
            ax.legend(loc='upper right')
            
            self.axes.append((ax, rest_line, left_line, right_line))
        
        self.fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, 
            interval=100,  # update every 100ms
            blit=False
        )
        
        # State windows
        self.state_windows = {}
        
        # Information labels
        self.info_frame = tk.Frame(self.root, bg='#202020')
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Text(
            self.info_frame,
            bg='#303030',
            fg='white',
            height=10,
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        self.info_text.pack(fill=tk.X, padx=20)
        
        self.info_text.insert(tk.END, """
EEG MOTOR IMAGERY PATTERN CHARACTERISTICS:

REST STATE: 
- Strong alpha rhythm (8-13 Hz) present in all motor cortex channels
- Amplitude is stable and rhythmic, especially in the central (Cz) channel
- Signal shows clear, regular oscillations with minimal event-related desynchronization

LEFT HAND MOTOR IMAGERY: 
- Decreased amplitude (event-related desynchronization) in RIGHT motor cortex (C4)
- Maintained or slightly increased amplitude in LEFT motor cortex (C3)
- Characteristic alpha/beta power decrease in contralateral hemisphere

RIGHT HAND MOTOR IMAGERY:
- Decreased amplitude (event-related desynchronization) in LEFT motor cortex (C3)
- Maintained or slightly increased amplitude in RIGHT motor cortex (C4)
- Mirror pattern of left hand motor imagery

Click on the buttons above to see detailed visualizations of each state.
        """)
        self.info_text.config(state=tk.DISABLED)
    
    def generate_eeg_signal(self, state, noise_level=0.5):
        """Generate simulated EEG for the given state with specific patterns"""
        # Base signal (alpha rhythm at 10Hz)
        base_alpha = np.sin(2 * np.pi * 10 * self.time_axis)
        base_beta = np.sin(2 * np.pi * 20 * self.time_axis) * 0.3
        
        # Add noise
        noise = np.random.normal(0, noise_level, self.buffer_size)
        
        # Different patterns for each channel and each state
        signals = []
        
        if state == 'rest':
            # REST: Strong alpha (10Hz) in all channels
            c3 = base_alpha * 1.5 + base_beta + noise * 0.4  # Strong alpha
            cz = base_alpha * 1.2 + base_beta + noise * 0.3  # Strong alpha
            c4 = base_alpha * 1.5 + base_beta + noise * 0.4  # Strong alpha
        
        elif state == 'left':
            # LEFT hand: Reduced alpha in right motor cortex (C4)
            c3 = base_alpha * 1.3 + base_beta + noise * 0.3  # Normal alpha
            cz = base_alpha * 1.0 + base_beta + noise * 0.4  # Slightly reduced alpha
            c4 = base_alpha * 0.6 + base_beta * 1.5 + noise * 0.8  # Strongly reduced alpha (ERD on right side)
        
        elif state == 'right':
            # RIGHT hand: Reduced alpha in left motor cortex (C3)
            c3 = base_alpha * 0.6 + base_beta * 1.5 + noise * 0.8  # Strongly reduced alpha (ERD on left side)
            cz = base_alpha * 1.0 + base_beta + noise * 0.4  # Slightly reduced alpha
            c4 = base_alpha * 1.3 + base_beta + noise * 0.3  # Normal alpha
        
        signals = [c3, cz, c4]
        return signals
    
    def update_plot(self, frame):
        """Update the main plot with all three states"""
        rest_signals = self.generate_eeg_signal('rest')
        left_signals = self.generate_eeg_signal('left')
        right_signals = self.generate_eeg_signal('right')
        
        for i, (ax, rest_line, left_line, right_line) in enumerate(self.axes):
            # Update data for each state
            rest_line.set_data(self.time_axis, rest_signals[i])
            left_line.set_data(self.time_axis, left_signals[i])
            right_line.set_data(self.time_axis, right_signals[i])
        
        return [line for _, rest_line, left_line, right_line in self.axes 
                for line in [rest_line, left_line, right_line]]
    
    def open_state_window(self, state):
        """Open a new window for detailed visualization of a specific state"""
        # Close existing window for this state if open
        if state in self.state_windows and self.state_windows[state]:
            self.state_windows[state].destroy()
        
        # Create new window
        window = tk.Toplevel(self.root)
        window.title(f"{state.upper()} State EEG Patterns")
        window.geometry("800x600")
        window.configure(bg='#202020')
        
        # Store reference to window
        self.state_windows[state] = window
        
        # Title
        title_label = tk.Label(
            window, 
            text=f"{state.upper()} State EEG Pattern Visualization",
            font=("Arial", 18, "bold"),
            bg='#202020', fg='white'
        )
        title_label.pack(pady=10)
        
        # Create figure
        fig = Figure(figsize=(8, 6), dpi=100)
        fig.patch.set_facecolor('#303030')
        
        # Create subplots for each channel
        state_axes = []
        for i in range(3):
            ax = fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('#303030')
            ax.set_title(self.channels[i], color='white', fontsize=12)
            ax.set_ylabel('Amplitude (μV)', color='white')
            ax.set_xlim(0, self.signal_length)
            ax.set_ylim(-2, 2)
            if i == 2:  # Only add x-label for the bottom plot
                ax.set_xlabel('Time (s)', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Color based on state
            color = '#00FF00' if state == 'rest' else '#2196F3' if state == 'left' else '#F44336'
            line, = ax.plot([], [], color=color, linewidth=2)
            
            state_axes.append((ax, line))
        
        fig.tight_layout(pad=3.0)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add annotations
        annotation_frame = tk.Frame(window, bg='#202020')
        annotation_frame.pack(fill=tk.X, pady=10)
        
        annotation_text = tk.Text(
            annotation_frame,
            bg='#303030',
            fg='white',
            height=6,
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        annotation_text.pack(fill=tk.X, padx=20)
        
        # Different annotations based on state
        if state == 'rest':
            annotation = """
REST STATE CHARACTERISTICS:
• Strong alpha (8-13 Hz) rhythm is present in all channels (C3, Cz, C4)
• Signal shows clear, regular oscillations with high amplitude
• Minimal event-related desynchronization (ERD)
• Most prominent in central (Cz) channel
• This pattern indicates no motor imagery activity (baseline state)
            """
        elif state == 'left':
            annotation = """
LEFT HAND MOTOR IMAGERY CHARACTERISTICS:
• Reduced alpha power (event-related desynchronization) in RIGHT motor cortex (C4)
• Maintained or slightly increased amplitude in LEFT motor cortex (C3)
• Most significant ERD occurs in beta band (13-30 Hz)
• This pattern reflects cortical activation in the contralateral hemisphere
• The right motor cortex shows decreased synchronization during left hand imagery
            """
        else:  # right
            annotation = """
RIGHT HAND MOTOR IMAGERY CHARACTERISTICS:
• Reduced alpha power (event-related desynchronization) in LEFT motor cortex (C3)
• Maintained or slightly increased amplitude in RIGHT motor cortex (C4)
• Mirror pattern of left hand motor imagery
• The left motor cortex shows decreased synchronization during right hand imagery
• This pattern is the neurophysiological basis for right hand movement classification
            """
        
        annotation_text.insert(tk.END, annotation)
        annotation_text.config(state=tk.DISABLED)
        
        # Set up animation
        def update_state_plot(frame):
            signals = self.generate_eeg_signal(state)
            for i, (ax, line) in enumerate(state_axes):
                line.set_data(self.time_axis, signals[i])
            return [line for _, line in state_axes]
        
        ani = animation.FuncAnimation(
            fig, update_state_plot, 
            interval=100,  # update every 100ms
            blit=False
        )
        
        # Keep reference to animation to prevent garbage collection
        window.ani = ani

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGStateVisualizer(root)
    root.mainloop()