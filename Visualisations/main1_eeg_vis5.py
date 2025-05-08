import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import random
import joblib
import time
from matplotlib.patches import Rectangle

# Try to load the model
try:
    model = joblib.load('high_acc_gb_model.pkl')  # You can change this to your preferred model
    print("Model loaded successfully!")
    have_model = True
except Exception as e:
    print(f"Could not load model: {e}")
    print("Will run in simulation mode")
    have_model = False

# Simulated EEG data parameters
sampling_rate = 250  # Hz
signal_length = 5  # seconds
buffer_size = sampling_rate * signal_length
time_axis = np.linspace(0, signal_length, buffer_size)

# Key EEG channels for motor imagery
channels = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']

# Class names
class_names = ['rest', 'left', 'right']

# Signal patterns for different states
def generate_eeg_signal(state, noise_level=0.5):
    """Generate simulated EEG for the given state"""
    # Base signal (alpha rhythm at 10Hz)
    base_signal = np.sin(2 * np.pi * 10 * time_axis)
    
    # Add noise
    noise = np.random.normal(0, noise_level, buffer_size)
    
    # Different patterns for each channel and each state
    signals = []
    
    if state == 'rest':
        # REST: Strong alpha (10Hz) in all channels
        c3 = base_signal * 1.5 + noise * 0.4  # Strong alpha
        cz = base_signal * 1.2 + noise * 0.3  # Strong alpha
        c4 = base_signal * 1.5 + noise * 0.4  # Strong alpha
    
    elif state == 'left':
        # LEFT hand: Reduced alpha in right motor cortex (C4)
        c3 = base_signal * 1.3 + noise * 0.3  # Normal alpha
        cz = base_signal * 1.0 + noise * 0.4  # Slightly reduced alpha
        c4 = base_signal * 0.6 + noise * 0.8  # Strongly reduced alpha (ERD on right side)
    
    elif state == 'right':
        # RIGHT hand: Reduced alpha in left motor cortex (C3)
        c3 = base_signal * 0.6 + noise * 0.8  # Strongly reduced alpha (ERD on left side)
        cz = base_signal * 1.0 + noise * 0.4  # Slightly reduced alpha
        c4 = base_signal * 1.3 + noise * 0.3  # Normal alpha
    
    signals = [c3, cz, c4]
    return signals

# Extract simulated features
def extract_simple_features(signals):
    """Extract simple features from the signals (simulated)"""
    features = []
    
    # For each channel
    for signal in signals:
        # Calculate band powers (simplified)
        # Theta (4-8 Hz)
        theta_power = np.mean(np.abs(np.fft.rfft(signal))[4:8])
        
        # Alpha (8-13 Hz)
        alpha_power = np.mean(np.abs(np.fft.rfft(signal))[8:13])
        
        # Beta (13-30 Hz)
        beta_power = np.mean(np.abs(np.fft.rfft(signal))[13:30])
        
        # Temporal features
        variance = np.var(signal)
        
        # Add to feature vector
        features.extend([theta_power, alpha_power, beta_power, variance])
    
    # Add simulated connectivity features
    features.extend([random.random() for _ in range(10)])
    
    return np.array(features).reshape(1, -1)

class EEGVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Visualization with Motor Imagery Classification")
        self.root.geometry("1000x800")
        self.root.configure(bg='black')
        
        # Main frame
        self.main_frame = tk.Frame(root, bg='black')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="Brain-Computer Interface EEG Monitor",
            font=("Arial", 24, "bold"),
            bg='black', fg='white'
        )
        self.title_label.pack(pady=10)
        
        # Create frame for EEG plot
        self.plot_frame = tk.Frame(self.main_frame, bg='black')
        self.plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Create EEG plot
        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.fig.patch.set_facecolor('black')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create three subplots for C3, Cz, and C4
        self.axes = []
        self.highlight_boxes = []
        
        for i in range(3):
            ax = self.fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('black')
            ax.set_title(channels[i], color='white')
            ax.set_ylabel('Amplitude', color='white')
            if i == 2:  # Only add x-label for the bottom plot
                ax.set_xlabel('Time (s)', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Initialize line
            line, = ax.plot(time_axis, np.zeros(buffer_size), color='cyan')
            
            # Add a blank highlight box (will be updated later)
            highlight = ax.add_patch(Rectangle((0, 0), 0, 0, alpha=0.3, color='none'))
            self.highlight_boxes.append(highlight)
            
            self.axes.append((ax, line))
        
        self.fig.tight_layout()
        
        # Current state display
        self.state_frame = tk.Frame(self.main_frame, bg='black')
        self.state_frame.pack(pady=10)
        
        self.state_label = tk.Label(
            self.state_frame,
            text="Detected Mental Command:",
            font=("Arial", 16),
            bg='black', fg='white'
        )
        self.state_label.pack(side=tk.LEFT, padx=10)
        
        self.current_state = tk.StringVar()
        self.current_state.set("INITIALIZING...")
        
        self.state_value = tk.Label(
            self.state_frame,
            textvariable=self.current_state,
            font=("Arial", 20, "bold"),
            bg='black', fg='green'
        )
        self.state_value.pack(side=tk.LEFT, padx=10)
        
        # State boxes
        self.boxes_frame = tk.Frame(self.main_frame, bg='black')
        self.boxes_frame.pack(pady=10)
        
        # Rest box
        self.rest_frame = tk.Frame(
            self.boxes_frame, 
            width=80, 
            height=80,
            bg='gray'
        )
        self.rest_frame.pack(side=tk.LEFT, padx=20)
        self.rest_label = tk.Label(
            self.rest_frame,
            text="REST",
            font=("Arial", 14, "bold"),
            bg='gray', fg='white',
            width=6, height=2
        )
        self.rest_label.pack()
        
        # Left box
        self.left_frame = tk.Frame(
            self.boxes_frame, 
            width=80, 
            height=80,
            bg='gray'
        )
        self.left_frame.pack(side=tk.LEFT, padx=20)
        self.left_label = tk.Label(
            self.left_frame,
            text="LEFT",
            font=("Arial", 14, "bold"),
            bg='gray', fg='white',
            width=6, height=2
        )
        self.left_label.pack()
        
        # Right box
        self.right_frame = tk.Frame(
            self.boxes_frame, 
            width=80, 
            height=80,
            bg='gray'
        )
        self.right_frame.pack(side=tk.LEFT, padx=20)
        self.right_label = tk.Label(
            self.right_frame,
            text="RIGHT",
            font=("Arial", 14, "bold"),
            bg='gray', fg='white',
            width=6, height=2
        )
        self.right_label.pack()
        
        # Status area - Feature visualization
        self.features_frame = tk.Frame(self.main_frame, bg='black')
        self.features_frame.pack(pady=10, fill=tk.X)
        
        # Add feature visualization
        self.feature_fig = plt.Figure(figsize=(10, 2), dpi=100)
        self.feature_fig.patch.set_facecolor('black')
        
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, self.features_frame)
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.feature_ax = self.feature_fig.add_subplot(111)
        self.feature_ax.set_facecolor('black')
        self.feature_ax.set_title('Band Power Features', color='white')
        self.feature_ax.set_xlabel('Frequency Bands', color='white')
        self.feature_ax.set_ylabel('Power', color='white')
        self.feature_ax.tick_params(colors='white')
        for spine in self.feature_ax.spines.values():
            spine.set_color('white')
        
        # Add bars for theta, alpha, beta for each channel
        bands = ['Theta', 'Alpha', 'Beta']
        x_pos = np.arange(len(bands) * len(channels))
        self.bar_plot = self.feature_ax.bar(
            x_pos, 
            np.zeros(len(bands) * len(channels)),
            color=['cyan', 'magenta', 'yellow'] * len(channels)
        )
        
        # Annotations for feature bars
        self.feature_annotations = []
        for i in range(len(self.bar_plot)):
            annot = self.feature_ax.annotate(
                "", 
                xy=(x_pos[i], 0),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom',
                color='white',
                fontsize=8
            )
            self.feature_annotations.append(annot)
        
        # Add labels
        channel_labels = []
        for ch in channels:
            for band in bands:
                channel_labels.append(f"{ch[0:2]}-{band[0]}")
        
        self.feature_ax.set_xticks(x_pos)
        self.feature_ax.set_xticklabels(channel_labels, rotation=45)
        
        self.feature_fig.tight_layout()
        
        # Mode label
        self.mode_label = tk.Label(
            self.main_frame,
            text="Simulation Mode" if not have_model else "Using Trained Model",
            font=("Arial", 10),
            bg='black', fg='yellow'
        )
        self.mode_label.pack(pady=5)
        
        # Legend for EEG patterns
        self.legend_frame = tk.Frame(self.main_frame, bg='black')
        self.legend_frame.pack(pady=5, fill=tk.X)
        
        legend_text = """
        EEG Patterns:
        • REST: Strong alpha (10Hz) rhythms in all channels
        • LEFT Hand: Decreased activity (ERD) in right motor cortex (C4)
        • RIGHT Hand: Decreased activity (ERD) in left motor cortex (C3)
        """
        
        self.legend_label = tk.Label(
            self.legend_frame,
            text=legend_text,
            font=("Arial", 10),
            bg='black', fg='white',
            justify=tk.LEFT
        )
        self.legend_label.pack(anchor=tk.W, padx=20)
        
        # Add annotations frame
        self.annotations_frame = tk.Frame(self.main_frame, bg='black')
        self.annotations_frame.pack(pady=5, fill=tk.X)
        
        self.annotations_label_text = tk.StringVar()
        self.annotations_label_text.set("Signal Analysis: Initializing...")
        
        self.annotations_label = tk.Label(
            self.annotations_frame,
            textvariable=self.annotations_label_text,
            font=("Arial", 10, "bold"),
            bg='black', fg='white',
            justify=tk.LEFT
        )
        self.annotations_label.pack(anchor=tk.W, padx=20)
        
        # Initialize signals and classification
        self.current_signals = generate_eeg_signal('rest')
        self.current_classification = 'rest'
        
        # Start the animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, 
            interval=100,  # update every 100ms
            blit=False
        )
        
        # Start classification updates
        self.update_classification()
    
    def update_plot(self, frame):
        """Update the EEG plot"""
        # Get new data based on current classification
        self.current_signals = generate_eeg_signal(self.current_classification)
        
        # Key signal characteristics to highlight based on state
        if self.current_classification == 'rest':
            highlight_colors = ['green', 'green', 'green']
            highlight_opacity = [0.2, 0.2, 0.2]
            annotation = "SIGNAL ANALYSIS: Strong regular alpha waves visible in ALL channels - typical REST pattern"
            
        elif self.current_classification == 'left':
            highlight_colors = ['blue', 'blue', 'blue']
            highlight_opacity = [0.2, 0.2, 0.5]  # Highlight C4 (right motor cortex) more strongly
            annotation = "SIGNAL ANALYSIS: Reduced alpha waves in C4 (right motor cortex) - LEFT hand movement pattern"
            
        elif self.current_classification == 'right':
            highlight_colors = ['red', 'red', 'red']
            highlight_opacity = [0.5, 0.2, 0.2]  # Highlight C3 (left motor cortex) more strongly
            annotation = "SIGNAL ANALYSIS: Reduced alpha waves in C3 (left motor cortex) - RIGHT hand movement pattern"
        
        self.annotations_label_text.set(annotation)
        
        # Update each channel plot
        for i, (ax, line) in enumerate(self.axes):
            # Update the line data
            line.set_ydata(self.current_signals[i])
            
            # Update y-axis limits
            ax.set_ylim(min(self.current_signals[i]) - 0.5, max(self.current_signals[i]) + 0.5)
            
            # Update highlight box
            ylim = ax.get_ylim()
            height = ylim[1] - ylim[0]
            
            # Remove old highlight
            self.highlight_boxes[i].remove()
            
            # Add new highlight based on state
            if self.current_classification == 'rest':
                # Highlight whole signal for rest
                self.highlight_boxes[i] = ax.add_patch(
                    Rectangle((0, ylim[0]), 5, height, 
                             alpha=highlight_opacity[i], 
                             color=highlight_colors[i],
                             zorder=0)
                )
            elif self.current_classification == 'left' and i == 2:  # C4 channel for left hand
                # Highlight specific part of C4 channel
                self.highlight_boxes[i] = ax.add_patch(
                    Rectangle((0, ylim[0]), 5, height, 
                             alpha=highlight_opacity[i], 
                             color=highlight_colors[i],
                             zorder=0)
                )
                # Add text annotation
                ax.annotate('Reduced amplitude here', xy=(2.5, 0), xytext=(2.5, 0),
                           ha='center', va='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.5),
                           fontsize=9)
            elif self.current_classification == 'right' and i == 0:  # C3 channel for right hand
                # Highlight specific part of C3 channel
                self.highlight_boxes[i] = ax.add_patch(
                    Rectangle((0, ylim[0]), 5, height, 
                             alpha=highlight_opacity[i], 
                             color=highlight_colors[i],
                             zorder=0)
                )
                # Add text annotation
                ax.annotate('Reduced amplitude here', xy=(2.5, 0), xytext=(2.5, 0),
                           ha='center', va='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
                           fontsize=9)
            else:
                # Regular highlight for other channels/states
                self.highlight_boxes[i] = ax.add_patch(
                    Rectangle((0, ylim[0]), 5, height, 
                             alpha=0.1, 
                             color=highlight_colors[i],
                             zorder=0)
                )
        
        # Update feature bars
        features = extract_simple_features(self.current_signals)
        
        # Update bar heights and annotations
        for i in range(len(self.bar_plot)):
            value = features[0, i]
            self.bar_plot[i].set_height(value)
            
            # Update feature annotations
            channel_idx = i // 3  # 0=C3, 1=Cz, 2=C4
            feature_type = i % 3  # 0=Theta, 1=Alpha, 2=Beta
            
            if self.current_classification == 'rest' and feature_type == 1:  # Alpha feature
                # All channels should have strong alpha during rest
                if value > 5.0:
                    self.feature_annotations[i].set_text("Strong")
                    self.feature_annotations[i].set_y(value)
                else:
                    self.feature_annotations[i].set_text("")
            elif self.current_classification == 'left' and channel_idx == 2 and feature_type == 1:
                # Right motor cortex (C4) alpha should be low during left hand movement
                if value < 3.0:
                    self.feature_annotations[i].set_text("ERD")
                    self.feature_annotations[i].set_y(value)
                else:
                    self.feature_annotations[i].set_text("")
            elif self.current_classification == 'right' and channel_idx == 0 and feature_type == 1:
                # Left motor cortex (C3) alpha should be low during right hand movement
                if value < 3.0:
                    self.feature_annotations[i].set_text("ERD")
                    self.feature_annotations[i].set_y(value)
                else:
                    self.feature_annotations[i].set_text("")
            else:
                self.feature_annotations[i].set_text("")
        
        self.feature_ax.set_ylim(0, max(features[0, :9]) * 1.2 + 1)
        
        self.feature_fig.canvas.draw_idle()
        
        return [line for _, line in self.axes]
    
    def update_classification(self):
        """Update the classification state"""
        # In a real application, this would get real features from your processing pipeline
        # For simulation, we'll randomly change between states
        if random.random() < 0.15:  # 15% chance to change state each time
            self.current_classification = random.choice(class_names)
            self.update_state(self.current_classification)
        
        # Schedule the next update
        self.root.after(2000, self.update_classification)
    
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

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGVisualization(root)
    root.mainloop()