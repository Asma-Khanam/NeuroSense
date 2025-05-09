import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import random
import joblib
import time
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import mplcursors
import os
import mne
from mne.datasets import eegbci
import threading

# Professional color scheme
COLORS = {
    'background': '#f0f0f0',      # Light gray background
    'panel_bg': '#ffffff',        # White panel background
    'text': '#333333',            # Dark gray text
    'primary': '#2196F3',         # Material Blue
    'secondary': '#4CAF50',       # Material Green
    'accent': '#FF9800',          # Material Orange
    'error': '#F44336',           # Material Red
    'grid': '#dddddd',            # Light gray grid
    'states': {
        'rest': '#4CAF50',        # Green
        'left': '#2196F3',        # Blue
        'right': '#FF9800'        # Orange
    },
    'highlight': {
        'rest': 'rgba(76, 175, 80, 0.2)',     # Light green
        'left': 'rgba(33, 150, 243, 0.2)',    # Light blue
        'right': 'rgba(255, 152, 0, 0.2)'     # Light orange
    }
}

# Try to load the model
try:
    model = joblib.load('models/high_acc_gb_model.pkl')  # You can change this to your preferred model
    print("Model loaded successfully!")
    have_model = True
except Exception as e:
    print(f"Could not load model: {e}")
    print("Will run in simulation mode")
    have_model = False

# Simulated EEG data parameters
sampling_rate = 250  # Hz
signal_length = 5    # seconds
buffer_size = sampling_rate * signal_length
time_axis = np.linspace(0, signal_length, buffer_size)

# Key EEG channels for motor imagery
default_channels = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']

# Class names
class_names = ['rest', 'left', 'right']

# Define runs for task 4 and task 5
task_runs = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

# Signal patterns for different states (for simulation mode)
def generate_eeg_signal(state, noise_level=0.5):
    """
    Generate simulated EEG for the given mental state.
    
    Parameters:
    -----------
    state : str
        Mental state ('rest', 'left', or 'right')
    noise_level : float
        Amount of noise to add (0.0-1.0)
        
    Returns:
    --------
    signals : list of ndarrays
        Simulated EEG signals for C3, Cz, and C4 channels
    """
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
    """
    Extract features from EEG signals for motor imagery classification.
    
    This function implements the following feature extraction methods:
    1. Band power calculation (theta, alpha, beta bands)
    2. Signal variance for each channel
    3. Simulated connectivity features
    
    Parameters:
    -----------
    signals : list of ndarray
        The EEG signal data for each channel
    
    Returns:
    --------
    features : ndarray, shape (1, n_features)
        The extracted feature vector ready for classification
    """
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

# Function to create a custom ttk styled frame with rounded corners
def create_rounded_frame(parent, **kwargs):
    """Create a frame with a rounded appearance using ttk styling"""
    frame = ttk.Frame(parent, **kwargs)
    return frame

class EEGVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain-Computer Interface | Motor Imagery Analysis")
        self.root.geometry("1200x900")
        
        # Configure the style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme
        
        # Configure custom styles
        self.style.configure('TFrame', background=COLORS['background'])
        self.style.configure('TLabel', background=COLORS['background'], foreground=COLORS['text'])
        self.style.configure('Header.TLabel', font=('Helvetica', 24, 'bold'), foreground=COLORS['primary'])
        self.style.configure('Subheader.TLabel', font=('Helvetica', 16), foreground=COLORS['text'])
        self.style.configure('StateBox.TFrame', borderwidth=2, relief='raised')
        self.style.configure('ActiveState.TFrame', borderwidth=2, relief='raised', background=COLORS['primary'])
        
        # Data mode
        self.use_real_data = False
        self.data_loaded = False
        
        # Real data storage
        self.subjects = [i for i in range(1, 11)]  # Subject IDs 1-10
        self.current_subject = 1
        self.current_task = 'Task4'
        self.current_run = 4
        self.raw_data = None
        self.epochs = None
        self.current_epoch = 0
        self.available_channels = []
        self.selected_channels = []
        self.fs = 160  # Default sampling rate
        
        # Main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="Brain-Computer Interface: Motor Imagery Analysis",
            style='Header.TLabel'
        )
        self.title_label.pack(pady=10)
        
        # Data selector panel
        self.data_selector_frame = create_rounded_frame(self.main_frame)
        self.data_selector_frame.pack(fill=tk.X, pady=5)
        
        # Data mode selector
        self.data_mode_label = ttk.Label(
            self.data_selector_frame,
            text="Data Mode:",
            font=("Helvetica", 12)
        )
        self.data_mode_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.data_mode_var = tk.StringVar(value="Simulation")
        self.data_mode_rb1 = ttk.Radiobutton(
            self.data_selector_frame,
            text="Simulation",
            variable=self.data_mode_var,
            value="Simulation",
            command=self.on_data_mode_change
        )
        self.data_mode_rb1.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.data_mode_rb2 = ttk.Radiobutton(
            self.data_selector_frame,
            text="Real Data",
            variable=self.data_mode_var,
            value="Real Data",
            command=self.on_data_mode_change
        )
        self.data_mode_rb2.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # Subject selector (initially disabled)
        self.subject_label = ttk.Label(
            self.data_selector_frame,
            text="Subject:",
            font=("Helvetica", 12)
        )
        self.subject_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.subject_var = tk.StringVar(value="1")
        self.subject_menu = ttk.Combobox(
            self.data_selector_frame,
            textvariable=self.subject_var,
            values=[str(i) for i in self.subjects],
            width=5,
            state='disabled'
        )
        self.subject_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.subject_menu.bind("<<ComboboxSelected>>", self.on_subject_change)
        
        # Task selector (initially disabled)
        self.task_label = ttk.Label(
            self.data_selector_frame,
            text="Task:",
            font=("Helvetica", 12)
        )
        self.task_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        
        self.task_var = tk.StringVar(value="Task4")
        self.task_menu = ttk.Combobox(
            self.data_selector_frame,
            textvariable=self.task_var,
            values=list(task_runs.keys()),
            width=8,
            state='disabled'
        )
        self.task_menu.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.task_menu.bind("<<ComboboxSelected>>", self.on_task_change)
        
        # Load button (initially disabled)
        self.load_button = ttk.Button(
            self.data_selector_frame,
            text="Load Data",
            command=self.load_data,
            state='disabled'
        )
        self.load_button.grid(row=1, column=4, padx=10, pady=5, sticky="w")
        
        # Channel selector (initially disabled)
        self.channel_label = ttk.Label(
            self.data_selector_frame,
            text="Channels:",
            font=("Helvetica", 12)
        )
        self.channel_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.channel_selector = ttk.Combobox(
            self.data_selector_frame,
            values=default_channels,
            width=25,
            state='disabled'
        )
        self.channel_selector.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        
        self.add_channel_button = ttk.Button(
            self.data_selector_frame,
            text="Add",
            command=self.add_channel,
            state='disabled'
        )
        self.add_channel_button.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        
        self.clear_channels_button = ttk.Button(
            self.data_selector_frame,
            text="Clear",
            command=self.clear_channels,
            state='disabled'
        )
        self.clear_channels_button.grid(row=2, column=4, padx=5, pady=5, sticky="w")
        
        # Selected channels display
        self.selected_channels_label = ttk.Label(
            self.data_selector_frame,
            text="Selected: C3, Cz, C4",
            font=("Helvetica", 10)
        )
        self.selected_channels_label.grid(row=3, column=0, columnspan=5, padx=10, pady=5, sticky="w")
        
        # Information bar
        self.info_frame = create_rounded_frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.current_state = tk.StringVar()
        self.current_state.set("INITIALIZING...")
        
        self.mode_label = ttk.Label(
            self.info_frame,
            text="Mode: " + ("Using Trained Model" if have_model else "Simulation Mode"),
            font=("Helvetica", 10),
        )
        self.mode_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.state_label = ttk.Label(
            self.info_frame,
            text="Mental Command:",
            font=("Helvetica", 12, "bold"),
        )
        self.state_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.state_value = ttk.Label(
            self.info_frame,
            textvariable=self.current_state,
            font=("Helvetica", 14, "bold"),
            foreground=COLORS['secondary']
        )
        self.state_value.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(
            self.info_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10)
        )
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Create tabs for different visualizations
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # EEG Signal Tab
        self.eeg_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eeg_tab, text="EEG Signals")
        
        # Feature Analysis Tab
        self.features_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.features_tab, text="Feature Analysis")
        
        # Brain Topography Tab (simulation)
        self.topo_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.topo_tab, text="Brain Topography")
        
        # Setup the EEG signal visualization
        self.setup_eeg_visualization()
        
        # Setup the feature visualization
        self.setup_feature_visualization()
        
        # Setup topography visualization
        self.setup_topography_visualization()
        
        # Control panel at the bottom
        self.control_frame = create_rounded_frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # Navigation controls (initially disabled)
        self.navigation_frame = ttk.Frame(self.control_frame)
        self.navigation_frame.pack(pady=10)
        
        self.prev_button = ttk.Button(
            self.navigation_frame,
            text="◄ Prev",
            command=self.prev_epoch,
            state='disabled'
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.playpause_button = ttk.Button(
            self.navigation_frame,
            text="❚❚ Pause",
            command=self.toggle_play_pause,
            state='disabled'
        )
        self.playpause_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(
            self.navigation_frame,
            text="Next ►",
            command=self.next_epoch,
            state='disabled'
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.epoch_var = tk.StringVar(value="Epoch: -/-")
        self.epoch_label = ttk.Label(
            self.navigation_frame,
            textvariable=self.epoch_var,
            font=("Helvetica", 10)
        )
        self.epoch_label.pack(side=tk.LEFT, padx=20)
        
        # State boxes
        self.states_frame = ttk.Frame(self.control_frame)
        self.states_frame.pack(pady=10)
        
        self.state_boxes = {}
        
        for i, state in enumerate(class_names):
            state_frame = ttk.Frame(self.states_frame, width=100, height=60, style='StateBox.TFrame')
            state_frame.pack(side=tk.LEFT, padx=15)
            state_frame.pack_propagate(False)  # Prevent the frame from shrinking
            
            state_label = ttk.Label(
                state_frame,
                text=state.upper(),
                font=("Helvetica", 14, "bold"),
                anchor="center",
            )
            state_label.pack(expand=True, fill=tk.BOTH)
            
            self.state_boxes[state] = (state_frame, state_label)
        
        # Status bar
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.StringVar()
        self.status_text.set("Ready. Monitoring EEG signals for motor imagery patterns.")
        
        self.status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_text,
            font=("Helvetica", 10),
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Initialize signals and classification
        self.current_signals = generate_eeg_signal('rest')
        self.current_classification = 'rest'
        self.paused = False
        
        # Initialize cache for efficient updates
        self._eeg_cache = {
            'current_state': None,
            'cached_signals': None,
            'cached_spectrogram': None,
            'y_limits': [None, None, None],
            'last_epoch': None
        }
        
        # Update animation creation with save_count parameter
        self.ani = animation.FuncAnimation(
            self.eeg_fig, self.update_eeg_plot, 
            interval=100,
            blit=False,
            save_count=50  # Only cache 50 frames
        )

        self.feature_ani = animation.FuncAnimation(
            self.feature_fig, self.update_feature_plots, 
            interval=200,
            blit=False,
            save_count=50
        )

        self.topo_ani = animation.FuncAnimation(
            self.topo_fig, self.update_topography, 
            interval=200,
            blit=False,
            save_count=50
        )
        
        # Start classification updates
        self.update_classification()
    
    def setup_eeg_visualization(self):
        """Create an enhanced EEG signal visualization panel"""
        # Frame for EEG plot
        self.eeg_plot_frame = ttk.Frame(self.eeg_tab)
        self.eeg_plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Create EEG plot with GridSpec for better layout
        self.eeg_fig = plt.Figure(figsize=(12, 7), dpi=100)
        self.eeg_gs = gridspec.GridSpec(3, 1)
        
        self.canvas = FigureCanvasTkAgg(self.eeg_fig, self.eeg_plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create three subplots for C3, Cz, and C4
        self.eeg_axes = []
        self.highlight_boxes = []
        self.erd_annotations = []
        
        # Custom colors for the lines
        channel_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        for i in range(3):
            ax = self.eeg_fig.add_subplot(self.eeg_gs[i])
            ax.set_facecolor(COLORS['panel_bg'])
            ax.set_title(default_channels[i], fontsize=12, color=COLORS['text'], fontweight='bold')
            ax.set_ylabel('Amplitude (μV)', fontsize=10, color=COLORS['text'])
            
            if i == 2:  # Only add x-label for the bottom plot
                ax.set_xlabel('Time (seconds)', fontsize=10, color=COLORS['text'])
            
            # Nicer grid
            ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
            
            # Better spine appearance
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
                spine.set_linewidth(0.5)
            
            # Initialize line with gradient color
            line, = ax.plot(
                time_axis, 
                np.zeros(buffer_size), 
                color=channel_colors[i],
                linewidth=1.5,
                alpha=0.8
            )
            
            # Add frequency band annotations
            ax.annotate(
                'Alpha band (8-13 Hz)',
                xy=(0.85, 0.9), 
                xycoords='axes fraction',
                fontsize=8,
                color=COLORS['text'],
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS['panel_bg'], ec=channel_colors[i], alpha=0.3)
            )
            
            # Add a blank highlight box (will be updated later)
            highlight = ax.add_patch(Rectangle((0, 0), 0, 0, alpha=0.2, color='none'))
            self.highlight_boxes.append(highlight)
            
            # Add an ERD annotation (initially invisible)
            erd_text = ax.text(
                0.5, 0.5, '', 
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                visible=False
            )
            self.erd_annotations.append(erd_text)
            
            self.eeg_axes.append((ax, line))
        
        # Add an explanation text widget instead of using a subplot
        explanation_label = ttk.Label(
            self.eeg_tab,
            text="EEG Motor Imagery Patterns:\n"
                "• REST: Strong alpha (10Hz) rhythms in all channels\n"
                "• LEFT Hand: Event-Related Desynchronization (ERD) in right motor cortex (C4)\n"
                "• RIGHT Hand: Event-Related Desynchronization (ERD) in left motor cortex (C3)",
            font=("Helvetica", 10),
            background=COLORS['background'],
            foreground=COLORS['text'],
            justify=tk.CENTER,
            padding=10
        )
        explanation_label.pack(pady=5, fill=tk.X)
        
        self.eeg_fig.tight_layout()
        
        # Add a time-frequency analysis panel
        self.tf_frame = ttk.Frame(self.eeg_tab)
        self.tf_frame.pack(pady=5, fill=tk.X)
        
        self.tf_label = ttk.Label(
            self.tf_frame,
            text="Time-Frequency Analysis",
            font=("Helvetica", 12, "bold"),
        )
        self.tf_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Time-frequency plot (simplified spectrogram simulation)
        self.tf_fig = plt.Figure(figsize=(12, 2), dpi=100)
        self.tf_canvas = FigureCanvasTkAgg(self.tf_fig, self.tf_frame)
        self.tf_canvas.get_tk_widget().pack(fill=tk.X, padx=10)
        
        self.tf_ax = self.tf_fig.add_subplot(111)
        self.tf_ax.set_facecolor(COLORS['panel_bg'])
        self.tf_ax.set_title('Spectrogram (C3 Channel)', fontsize=10, color=COLORS['text'])
        self.tf_ax.set_ylabel('Frequency (Hz)', fontsize=9, color=COLORS['text'])
        self.tf_ax.set_xlabel('Time (s)', fontsize=9, color=COLORS['text'])
        
        # Create a simulated spectrogram (will be updated in the animation)
        self.spectrogram = self.tf_ax.imshow(
            np.random.rand(30, 100),
            aspect='auto',
            origin='lower',
            extent=[0, 5, 0, 30],
            cmap='viridis',
            interpolation='bilinear'
        )
        
        # Add frequency band markers
        self.tf_ax.axhline(y=4, color='white', linestyle='--', alpha=0.5)
        self.tf_ax.axhline(y=8, color='white', linestyle='--', alpha=0.5)
        self.tf_ax.axhline(y=13, color='white', linestyle='--', alpha=0.5)
        self.tf_ax.axhline(y=30, color='white', linestyle='--', alpha=0.5)
        
        # Add band labels
        self.tf_ax.text(5.1, 2, 'Delta', fontsize=8, ha='left', va='center', color='white')
        self.tf_ax.text(5.1, 6, 'Theta', fontsize=8, ha='left', va='center', color='white')
        self.tf_ax.text(5.1, 10, 'Alpha', fontsize=8, ha='left', va='center', color='white')
        self.tf_ax.text(5.1, 20, 'Beta', fontsize=8, ha='left', va='center', color='white')
        
        # Add colorbar
        cbar = self.tf_fig.colorbar(self.spectrogram, ax=self.tf_ax, shrink=0.8)
        cbar.set_label('Power (μV²/Hz)', fontsize=8, color=COLORS['text'])
        
        self.tf_fig.tight_layout()
    
    def setup_feature_visualization(self):
        """Create enhanced feature visualization panel"""
        # Create a layout with multiple feature visualizations
        self.feature_frame = ttk.Frame(self.features_tab)
        self.feature_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Main title for the feature analysis
        self.feature_title = ttk.Label(
            self.feature_frame,
            text="EEG Feature Analysis",
            font=("Helvetica", 14, "bold"),
        )
        self.feature_title.pack(pady=5)
        
        # Create a figure with multiple subplots for different feature visualizations
        self.feature_fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, self.feature_frame)
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up grid for multiple visualizations
        gs = gridspec.GridSpec(2, 2, figure=self.feature_fig)
        
        # 1. Radar chart for band powers
        self.radar_ax = self.feature_fig.add_subplot(gs[0, 0], polar=True)
        self.radar_ax.set_facecolor(COLORS['panel_bg'])
        self.radar_ax.set_title('EEG Band Power Distribution', fontsize=12, color=COLORS['text'])
        
        # Create radar chart categories and initial data
        self.radar_categories = ['C3-Theta', 'C3-Alpha', 'C3-Beta', 
                                'Cz-Theta', 'Cz-Alpha', 'Cz-Beta',
                                'C4-Theta', 'C4-Alpha', 'C4-Beta']
        
        # Set up the angles for the radar chart
        self.radar_num_vars = len(self.radar_categories)
        self.radar_angles = np.linspace(0, 2*np.pi, self.radar_num_vars, endpoint=False).tolist()
        self.radar_angles += self.radar_angles[:1]  # Close the loop
        
        # Set radar chart labels
        self.radar_ax.set_xticks(self.radar_angles[:-1])
        self.radar_ax.set_xticklabels(self.radar_categories, color=COLORS['text'], fontsize=8)
        
        # Initial radar data (will be updated)
        self.radar_values = [0] * self.radar_num_vars
        self.radar_values += self.radar_values[:1]  # Close the loop
        
        # Plot radar chart
        self.radar_line, = self.radar_ax.plot(self.radar_angles, self.radar_values, 
                                              linewidth=2, linestyle='solid', 
                                              color=COLORS['primary'])
        self.radar_ax.fill(self.radar_angles, self.radar_values, 
                           color=COLORS['primary'], alpha=0.25)
        
        # 2. Bar chart for band powers
        self.bar_ax = self.feature_fig.add_subplot(gs[0, 1])
        self.bar_ax.set_facecolor(COLORS['panel_bg'])
        self.bar_ax.set_title('Band Power Comparison', fontsize=12, color=COLORS['text'])
        self.bar_ax.set_xlabel('Channel-Band', fontsize=10, color=COLORS['text'])
        self.bar_ax.set_ylabel('Power', fontsize=10, color=COLORS['text'])
        
        # Create bar positions
        self.bar_x = np.arange(len(self.radar_categories))
        
        # Initial empty bars
        self.bars = self.bar_ax.bar(
            self.bar_x, 
            np.zeros(len(self.radar_categories)),
            color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']] * 3,
            alpha=0.7
        )
        
        # Set bar chart labels
        self.bar_ax.set_xticks(self.bar_x)
        self.bar_ax.set_xticklabels(self.radar_categories, rotation=45, ha='right', fontsize=8, color=COLORS['text'])
        
        # Add grid
        self.bar_ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        # 3. Time series of feature changes
        self.timeseries_ax = self.feature_fig.add_subplot(gs[1, 0])
        self.timeseries_ax.set_facecolor(COLORS['panel_bg'])
        self.timeseries_ax.set_title('Alpha Power Over Time', fontsize=12, color=COLORS['text'])
        self.timeseries_ax.set_xlabel('Time (last 10 seconds)', fontsize=10, color=COLORS['text'])
        self.timeseries_ax.set_ylabel('Alpha Power', fontsize=10, color=COLORS['text'])
        
        # Initialize time series data buffers
        self.ts_buffer_size = 100  # 10 seconds at 10Hz update rate
        self.ts_time = np.linspace(-10, 0, self.ts_buffer_size)
        self.ts_c3_data = np.zeros(self.ts_buffer_size)
        self.ts_cz_data = np.zeros(self.ts_buffer_size)
        self.ts_c4_data = np.zeros(self.ts_buffer_size)
        
        # Plot time series
        self.ts_c3_line, = self.timeseries_ax.plot(self.ts_time, self.ts_c3_data, 
                                                   linewidth=1.5, color=COLORS['primary'], 
                                                   label='C3 (Left)')
        self.ts_cz_line, = self.timeseries_ax.plot(self.ts_time, self.ts_cz_data, 
                                                  linewidth=1.5, color=COLORS['secondary'], 
                                                  label='Cz (Center)')
        self.ts_c4_line, = self.timeseries_ax.plot(self.ts_time, self.ts_c4_data, 
                                                   linewidth=1.5, color=COLORS['accent'], 
                                                   label='C4 (Right)')
        
        # Add legend
        self.timeseries_ax.legend(fontsize=8)
        
        # Add grid
        self.timeseries_ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        # 4. Feature description/ERD visualization
        self.erd_ax = self.feature_fig.add_subplot(gs[1, 1])
        self.erd_ax.set_facecolor(COLORS['panel_bg'])
        self.erd_ax.set_title('Event-Related Desynchronization (ERD)', fontsize=12, color=COLORS['text'])
        
        # Create a schematic of ERD
        self.erd_x = np.linspace(0, 5, 500)
        self.baseline = np.sin(2*np.pi*10*self.erd_x) * 0.8 + np.random.normal(0, 0.1, 500)
        self.erd_signal = np.copy(self.baseline)
        
        # Create ERD in the middle
        erd_idx = np.logical_and(self.erd_x >= 2, self.erd_x <= 3)
        self.erd_signal[erd_idx] = self.erd_signal[erd_idx] * 0.4  # Reduce amplitude during ERD
        
        # Plot ERD demonstration
        self.erd_ax.plot(self.erd_x, self.baseline, color='gray', alpha=0.5, linewidth=1, label='Baseline')
        self.erd_ax.plot(self.erd_x, self.erd_signal, color=COLORS['primary'], linewidth=1.5, label='ERD')
        
        # Highlight ERD period
        self.erd_ax.axvspan(2, 3, alpha=0.2, color=COLORS['primary'])
        self.erd_ax.text(2.5, 0, 'ERD', ha='center', fontsize=10, color=COLORS['text'],
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=COLORS['primary'], alpha=0.7))
        
        # Add description text
        self.erd_ax.text(0.5, -1.5, 
                       "ERD is a decrease in neural oscillations (particularly in the alpha band)\n"
                       "when a brain region becomes active during motor imagery or actual movement.",
                       ha='center', fontsize=8, color=COLORS['text'])
        
        # Add legend
        self.erd_ax.legend(fontsize=8)
        
        # Set axis limits
        self.erd_ax.set_xlim(0, 5)
        self.erd_ax.set_ylim(-2, 2)
        
        # Remove y ticks for cleaner appearance
        self.erd_ax.set_yticks([])
        
        # Add x label
        self.erd_ax.set_xlabel('Time (s)', fontsize=10, color=COLORS['text'])
        
        # Adjust layout
        self.feature_fig.tight_layout()
    
    def setup_topography_visualization(self):
        """Create a simulated brain topography visualization"""
        # Create frame for topography
        self.topo_frame = ttk.Frame(self.topo_tab)
        self.topo_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Title for the topography section
        self.topo_title = ttk.Label(
            self.topo_frame,
            text="Brain Activity Topography",
            font=("Helvetica", 14, "bold"),
        )
        self.topo_title.pack(pady=5)
        
        # Create a figure for the topography
        self.topo_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.topo_canvas = FigureCanvasTkAgg(self.topo_fig, self.topo_frame)
        self.topo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create a grid for different views
        topo_gs = gridspec.GridSpec(2, 2, figure=self.topo_fig)
        
        # Top view
        self.topo_top_ax = self.topo_fig.add_subplot(topo_gs[0, 0])
        self.topo_top_ax.set_facecolor(COLORS['panel_bg'])
        self.topo_top_ax.set_title('Top View', fontsize=12, color=COLORS['text'])
        
        # Side view
        self.topo_side_ax = self.topo_fig.add_subplot(topo_gs[0, 1])
        self.topo_side_ax.set_facecolor(COLORS['panel_bg'])
        self.topo_side_ax.set_title('Side View', fontsize=12, color=COLORS['text'])
        
        # Alpha band topography
        self.topo_alpha_ax = self.topo_fig.add_subplot(topo_gs[1, 0])
        self.topo_alpha_ax.set_facecolor(COLORS['panel_bg'])
        self.topo_alpha_ax.set_title('Alpha Band (8-13 Hz)', fontsize=12, color=COLORS['text'])
        
        # Beta band topography
        self.topo_beta_ax = self.topo_fig.add_subplot(topo_gs[1, 1])
        self.topo_beta_ax.set_facecolor(COLORS['panel_bg'])
        self.topo_beta_ax.set_title('Beta Band (13-30 Hz)', fontsize=12, color=COLORS['text'])
        
        # Create a simulated head outline for each view
        self.draw_head_outline(self.topo_top_ax, view='top')
        self.draw_head_outline(self.topo_side_ax, view='side')
        self.draw_head_outline(self.topo_alpha_ax, view='top')
        self.draw_head_outline(self.topo_beta_ax, view='top')
        
        # Mark electrode positions
        self.mark_electrodes(self.topo_top_ax, view='top')
        self.mark_electrodes(self.topo_side_ax, view='side')
        
        # Initialize heatmaps for alpha and beta bands
        self.alpha_heatmap = self.create_topo_heatmap(self.topo_alpha_ax)
        self.beta_heatmap = self.create_topo_heatmap(self.topo_beta_ax)
        
        # Adjust layout
        self.topo_fig.tight_layout()
        
        # Add explanation text
        self.topo_explanation = ttk.Label(
            self.topo_frame,
            text=(
                "Brain topography shows the spatial distribution of EEG activity across the scalp.\n"
                "For motor imagery, we focus on activity over the sensorimotor cortex (areas C3, Cz, and C4).\n"
                "RED areas indicate higher activity, BLUE areas indicate lower activity (event-related desynchronization)."
            ),
            font=("Helvetica", 10),
            justify=tk.CENTER,
        )
        self.topo_explanation.pack(pady=10)
    
    def draw_head_outline(self, ax, view='top'):
        """Draw a simple head outline on the given axis"""
        if view == 'top':
            # Draw head circle
            circle = plt.Circle((0, 0), 1, fill=False, color=COLORS['text'], linewidth=2)
            ax.add_patch(circle)
            
            # Draw nose
            ax.plot([0, 0], [0.9, 1.1], color=COLORS['text'], linewidth=2)
            
            # Draw ears
            ax.plot([-1.1, -0.9], [0, 0], color=COLORS['text'], linewidth=2)
            ax.plot([0.9, 1.1], [0, 0], color=COLORS['text'], linewidth=2)
            
        elif view == 'side':
            # Draw head profile
            x = np.linspace(-1, 1, 100)
            y_top = np.sqrt(1 - x**2)
            
            # Head outline
            ax.plot(x, y_top, color=COLORS['text'], linewidth=2)
            ax.plot([-1, 1], [0, 0], color=COLORS['text'], linewidth=2)
            
            # Nose
            ax.plot([0.9, 1.1], [0.2, 0.2], color=COLORS['text'], linewidth=2)
            
            # Ear
            ax.plot([-1.05, -1.05], [-0.2, 0.2], color=COLORS['text'], linewidth=2)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Set limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    
    def mark_electrodes(self, ax, view='top'):
        """Mark electrode positions on the given axis"""
        if view == 'top':
            # Define positions for key electrodes (simplified)
            positions = {
                'C3': (-0.5, 0),
                'Cz': (0, 0),
                'C4': (0.5, 0),
                'F3': (-0.5, 0.5),
                'Fz': (0, 0.5),
                'F4': (0.5, 0.5),
                'P3': (-0.5, -0.5),
                'Pz': (0, -0.5),
                'P4': (0.5, -0.5)
            }
            
            # Mark each electrode
            for name, (x, y) in positions.items():
                ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
                ax.text(x, y+0.1, name, ha='center', va='center', fontsize=8, color=COLORS['text'])
                
                # Highlight motor cortex electrodes
                if name in ['C3', 'Cz', 'C4']:
                    ax.plot(x, y, 'o', markersize=10, mfc='none', mec=COLORS['accent'], linewidth=2)
        
        elif view == 'side':
            # Define positions for side view (simplified)
            positions = {
                'C3': (-0.5, 0.75),
                'Cz': (0, 1),
                'C4': (0.5, 0.75),
                'F3': (-0.75, 0.5),
                'Fz': (0, 0.75),
                'F4': (0.75, 0.5),
            }
            
            # Mark each electrode
            for name, (x, y) in positions.items():
                if x >= -0.2:  # Only show electrodes visible from this side
                    ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
                    ax.text(x, y+0.1, name, ha='center', va='center', fontsize=8, color=COLORS['text'])
                    
                    # Highlight motor cortex electrodes
                    if name in ['Cz', 'C4']:
                        ax.plot(x, y, 'o', markersize=10, mfc='none', mec=COLORS['accent'], linewidth=2)
    
    def create_topo_heatmap(self, ax):
        """Create a topographic heatmap on the given axis"""
        # Create a grid for the heatmap
        n = 20
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        
        # Create a circular mask
        mask = X**2 + Y**2 <= 1
        
        # Initialize data (will be updated in animation)
        data = np.zeros((n, n))
        data[mask] = np.random.rand(np.sum(mask))
        
        # Create a custom colormap (blue=low, red=high)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('ERD_ERS', colors, N=100)
        
        # Plot the heatmap
        im = ax.imshow(
            data, 
            extent=[-1.2, 1.2, -1.2, 1.2],
            origin='lower',
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation='bilinear'
        )
        
        # Add colorbar to the correct figure
        cbar = self.topo_fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Activity', fontsize=8, color=COLORS['text'])
        
        return im
    
    def update_eeg_plot(self, frame):
        """Update the EEG plot with maximum performance optimizations"""
        # Early return if this tab isn't currently visible - major performance boost
        if self.notebook.index(self.notebook.select()) != 0:
            return []
        
        # SIMULATION MODE
        if not self.use_real_data or not self.data_loaded:
            # Check if state has changed since last update
            if self._eeg_cache['current_state'] != self.current_classification:
                # State changed - generate new signals
                self.current_signals = generate_eeg_signal(self.current_classification)
                self._eeg_cache['cached_signals'] = self.current_signals
                self._eeg_cache['current_state'] = self.current_classification
                
                # Pre-calculate spectrogram data for the new state
                spec_data = np.zeros((30, 100))
                alpha_idx = slice(8, 13)
                beta_idx = slice(13, 30)
                theta_idx = slice(4, 8)
                
                # Add state-specific patterns (vectorized operations)
                if self.current_classification == 'rest':
                    # Strong alpha in all channels
                    spec_data[alpha_idx, :] = 0.8 + np.random.rand(5, 100) * 0.2
                    highlight_colors = [COLORS['states']['rest']] * 3
                    highlight_opacity = [0.2] * 3
                    annotation = "Strong regular alpha waves visible in ALL channels - typical REST pattern"
                    
                elif self.current_classification == 'left':
                    # Reduced alpha in right motor cortex (C4)
                    spec_data[alpha_idx, :] = 0.8 + np.random.rand(5, 100) * 0.2
                    # Add ERD in the middle portion for C4 (vectorized)
                    spec_data[alpha_idx, 40:70] = 0.3 + np.random.rand(5, 30) * 0.2
                    
                    highlight_colors = [COLORS['states']['left']] * 3
                    highlight_opacity = [0.1, 0.1, 0.4]  # Highlight C4 (right motor cortex) more strongly
                    annotation = "Reduced alpha waves in C4 (right motor cortex) - LEFT hand movement pattern"
                    
                elif self.current_classification == 'right':
                    # Reduced alpha in left motor cortex (C3)
                    spec_data[alpha_idx, :] = 0.8 + np.random.rand(5, 100) * 0.2
                    # Add ERD in the middle portion for C3 (vectorized)
                    spec_data[alpha_idx, 40:70] = 0.3 + np.random.rand(5, 30) * 0.2
                    
                    highlight_colors = [COLORS['states']['right']] * 3
                    highlight_opacity = [0.4, 0.1, 0.1]  # Highlight C3 (left motor cortex) more strongly
                    annotation = "Reduced alpha waves in C3 (left motor cortex) - RIGHT hand movement pattern"
                
                # Add frequency band activity (vectorized)
                spec_data[beta_idx, :] = 0.3 + np.random.rand(beta_idx.stop - beta_idx.start, 100) * 0.2
                spec_data[theta_idx, :] = 0.5 + np.random.rand(4, 100) * 0.2
                
                # Cache these for reuse
                self._eeg_cache['cached_spectrogram'] = spec_data
                self._eeg_cache['highlight_colors'] = highlight_colors
                self._eeg_cache['highlight_opacity'] = highlight_opacity
                
                # Update spectrogram and annotation immediately
                self.spectrogram.set_array(spec_data)
                self.status_text.set(annotation)
                
            # Reuse cached signals if state hasn't changed
            signals = self._eeg_cache['cached_signals']
            
            # Update channel plots efficiently - only once per state change to prevent redundant calculations
            for i, (ax, line) in enumerate(self.eeg_axes):
                # Update the line data
                line.set_ydata(signals[i])
                
                # Check if y-limits need updating
                data_min, data_max = signals[i].min(), signals[i].max()
                
                if self._eeg_cache['y_limits'][i] is None or \
                data_min < self._eeg_cache['y_limits'][i][0] or \
                data_max > self._eeg_cache['y_limits'][i][1]:
                    
                    padding = 0.5  # Fixed padding for stability
                    new_ylim = (data_min - padding, data_max + padding)
                    ax.set_ylim(new_ylim)
                    self._eeg_cache['y_limits'][i] = new_ylim
                    
                    # When y-limits change, we need to update the highlight boxes
                    # Remove old highlight
                    if hasattr(self, 'highlight_boxes') and i < len(self.highlight_boxes):
                        self.highlight_boxes[i].remove()
                    
                    # Get highlight properties from cache
                    highlight_colors = self._eeg_cache.get('highlight_colors', [COLORS['states']['rest']] * 3)
                    highlight_opacity = self._eeg_cache.get('highlight_opacity', [0.2] * 3)
                    
                    # Get needed dimensions for the highlight
                    ylim = ax.get_ylim()
                    height = ylim[1] - ylim[0]
                    
                    # Reset ERD annotations
                    if hasattr(self, 'erd_annotations') and i < len(self.erd_annotations):
                        self.erd_annotations[i].set_visible(False)
                    
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
                        # Make ERD annotation visible
                        self.erd_annotations[i].set_visible(True)
                        self.erd_annotations[i].set_text('ERD (Event-Related Desynchronization)')
                        self.erd_annotations[i].set_bbox(dict(
                            boxstyle='round,pad=0.5', 
                            fc=COLORS['states']['left'], 
                            alpha=0.2, 
                            ec=COLORS['states']['left']
                        ))
                    elif self.current_classification == 'right' and i == 0:  # C3 channel for right hand
                        # Highlight specific part of C3 channel
                        self.highlight_boxes[i] = ax.add_patch(
                            Rectangle((0, ylim[0]), 5, height, 
                                    alpha=highlight_opacity[i], 
                                    color=highlight_colors[i],
                                    zorder=0)
                        )
                        # Make ERD annotation visible
                        self.erd_annotations[i].set_visible(True)
                        self.erd_annotations[i].set_text('ERD (Event-Related Desynchronization)')
                        self.erd_annotations[i].set_bbox(dict(
                            boxstyle='round,pad=0.5', 
                            fc=COLORS['states']['right'], 
                            alpha=0.2, 
                            ec=COLORS['states']['right']
                        ))
                    else:
                        # Regular highlight for other channels/states
                        self.highlight_boxes[i] = ax.add_patch(
                            Rectangle((0, ylim[0]), 5, height, 
                                    alpha=0.1, 
                                    color=highlight_colors[i],
                                    zorder=0)
                        )
        
        # REAL DATA MODE
        elif self.data_loaded and self.use_real_data and not self.paused:
            # Check if we need to update data (new epoch or first load)
            if self.epochs is not None and self.current_epoch < len(self.epochs) and \
            (self._eeg_cache['last_epoch'] != self.current_epoch or self._eeg_cache['cached_signals'] is None):
                try:
                    # Get indices of selected channels - do this once for efficiency
                    if not hasattr(self, '_channel_indices') or self._channel_indices is None:
                        self._channel_indices = []
                        for ch_name in self.selected_channels:
                            if ch_name in self.channel_names:
                                self._channel_indices.append(self.channel_names.index(ch_name))
                    
                    # Check if we have valid channels
                    if not self._channel_indices:
                        return [line for _, line in self.eeg_axes]
                    
                    # Memory-efficient approach: only load data for this specific epoch and selected channels
                    try:
                        # Fastest approach for MNE
                        epoch_data = self.epochs.get_data(
                            picks=self._channel_indices[:3],  # Get at most 3 channels
                            item=self.current_epoch,
                            verbose=False
                        )
                    except:
                        # Fallback: get all epochs data and extract what we need
                        # Less efficient but more compatible
                        all_data = self.epochs[self.current_epoch].get_data()
                        epoch_data = np.array([all_data[0, i] for i in self._channel_indices[:3]])
                    
                    # Cache the results
                    self._eeg_cache['cached_signals'] = epoch_data
                    self._eeg_cache['last_epoch'] = self.current_epoch
                    
                    # Reset y-limits for new data
                    self._eeg_cache['y_limits'] = [None, None, None]
                    
                    # Compute spectrogram when epoch changes (only if we have data)
                    if len(epoch_data) > 0:
                        try:
                            from scipy import signal as sg
                            signal = epoch_data[0]  # Use first channel for spectrogram
                            
                            # Efficient spectrogram calculation with minimal parameters
                            nperseg = min(64, len(signal)//8)  # Balance between speed and quality
                            f, t, Sxx = sg.spectrogram(
                                signal, 
                                fs=self.fs, 
                                nperseg=nperseg,
                                noverlap=nperseg//4,  # 75% overlap
                                scaling='density', 
                                mode='magnitude'
                            )
                            
                            # Trim frequency to 0-30Hz for display
                            freq_mask = f <= 30
                            f = f[freq_mask]
                            Sxx = Sxx[freq_mask]
                            
                            # Resize for display
                            from scipy.ndimage import zoom
                            
                            # Target dimensions for display
                            target_freq = 30
                            target_time = 100
                            
                            # Compute zoom factors - avoid values < 1 which cause blurring
                            freq_factor = max(1, target_freq / Sxx.shape[0])
                            time_factor = max(1, target_time / Sxx.shape[1])
                            
                            # Fast resize with nearest-neighbor interpolation
                            if freq_factor != 1 or time_factor != 1:
                                Sxx_resized = zoom(Sxx, (freq_factor, time_factor), order=0)
                            else:
                                Sxx_resized = Sxx
                            
                            # Crop to exact size if needed
                            if Sxx_resized.shape[0] > target_freq:
                                Sxx_resized = Sxx_resized[:target_freq]
                            
                            if Sxx_resized.shape[1] > target_time:
                                Sxx_resized = Sxx_resized[:, :target_time]
                            
                            # Normalize for display
                            Sxx_max = Sxx_resized.max()
                            if Sxx_max > 0:
                                Sxx_resized = Sxx_resized / Sxx_max
                            
                            # Update the spectrogram
                            spec_data = np.zeros((30, 100))
                            h, w = min(Sxx_resized.shape[0], 30), min(Sxx_resized.shape[1], 100)
                            spec_data[:h, :w] = Sxx_resized[:h, :w]
                            
                            self._eeg_cache['cached_spectrogram'] = spec_data
                            self.spectrogram.set_array(spec_data)
                        
                        except Exception as e:
                            print(f"Error computing spectrogram: {e}")
                    
                    # Update UI with data epoch info
                    self.epoch_var.set(f"Epoch: {self.current_epoch + 1}/{len(self.epochs)}")
                    
                    # Try to get event info for this epoch
                    try:
                        # Identify the actual state/class for this epoch
                        event_id = self.epochs.events[self.current_epoch, 2]
                        
                        # Convert to state name
                        event_dict = {v: k for k, v in self.epochs.event_id.items()}
                        if event_id in event_dict:
                            state = event_dict[event_id]
                            
                            # Map to our standard states if necessary
                            if state == 'hands':
                                state = 'left'
                            elif state == 'feet':
                                state = 'right'
                            
                            # Ensure it's one of our recognized states
                            if state not in class_names:
                                state = 'rest'
                                
                            self.update_state(state)
                            
                            # Update annotation based on state
                            if state == 'rest':
                                annotation = "REST state: Strong alpha waves visible in all channels"
                            elif state == 'left':
                                annotation = "LEFT hand imagery: ERD visible in right motor cortex (C4)"
                            elif state == 'right':
                                annotation = "RIGHT hand imagery: ERD visible in left motor cortex (C3)"
                            else:
                                annotation = f"Current state: {state.upper()}"
                                
                            self.status_text.set(annotation)
                    except Exception as e:
                        print(f"Error identifying state: {e}")
                
                except Exception as e:
                    print(f"Error loading epoch data: {e}")
                    return [line for _, line in self.eeg_axes]
            
            # Use cached data for display updates
            signals = self._eeg_cache['cached_signals']
            
            if signals is not None:
                # Update plots for real data
                for i, (ax, line) in enumerate(self.eeg_axes[:len(signals)]):
                    # Update channel name in title
                    if i < len(self.selected_channels):
                        ax.set_title(self.selected_channels[i], fontsize=12, color=COLORS['text'], fontweight='bold')
                    
                    # Update line data
                    line.set_ydata(signals[i])
                    
                    # Update x-axis if needed
                    if line.get_xdata().size != len(self.time_axis):
                        line.set_xdata(self.time_axis)
                        ax.set_xlim(min(self.time_axis), max(self.time_axis))
                    
                    # Check if y-limits need updating to avoid excessive redrawing
                    data_min, data_max = signals[i].min(), signals[i].max()
                    
                    if self._eeg_cache['y_limits'][i] is None or \
                    data_min < self._eeg_cache['y_limits'][i][0] or \
                    data_max > self._eeg_cache['y_limits'][i][1]:
                        
                        # Add padding proportional to data range
                        padding = (data_max - data_min) * 0.1 if data_max > data_min else 5.0
                        new_ylim = (data_min - padding, data_max + padding)
                        ax.set_ylim(new_ylim)
                        self._eeg_cache['y_limits'][i] = new_ylim
        
        # Return the updated artists for animation
        return [line for _, line in self.eeg_axes]
    
    def update_feature_plots(self, frame):
        """Update the feature visualizations with error handling"""
        if not self.use_real_data:
            # Extract features from current signals
            features = extract_simple_features(self.current_signals)
            
            # Update radar chart data - first 9 features are band powers
            radar_data = features[0, :9]
            
            # Update radar chart
            self.radar_values = radar_data.tolist()
            self.radar_values += self.radar_values[:1]  # Close the loop
            self.radar_line.set_ydata(self.radar_values)
            
            # Safely update the radar chart fill
            try:
                # Try to remove existing fill if it exists
                if len(self.radar_ax.collections) > 0:
                    self.radar_ax.collections[0].remove()
                # Add new fill
                self.radar_ax.fill(self.radar_angles, self.radar_values, 
                                 color=COLORS['primary'], alpha=0.25)
            except Exception as e:
                print(f"Warning: Radar chart update error: {e}")
            
            # Update bar heights and annotations
            for i in range(len(self.bars)):
                value = radar_data[i]
                self.bars[i].set_height(value)
            
            # Determine max value for axis limits
            max_value = max(radar_data) * 1.2
            self.radar_ax.set_ylim(0, max_value)
            self.bar_ax.set_ylim(0, max_value)
            
            # Update time series data
            # Shift existing data to the left
            self.ts_c3_data = np.roll(self.ts_c3_data, -1)
            self.ts_cz_data = np.roll(self.ts_cz_data, -1)
            self.ts_c4_data = np.roll(self.ts_c4_data, -1)
            
            # Add new data points (alpha band power)
            self.ts_c3_data[-1] = features[0, 1]  # C3 Alpha
            self.ts_cz_data[-1] = features[0, 4]  # Cz Alpha
            self.ts_c4_data[-1] = features[0, 7]  # C4 Alpha
            
            # Update lines
            self.ts_c3_line.set_ydata(self.ts_c3_data)
            self.ts_cz_line.set_ydata(self.ts_cz_data)
            self.ts_c4_line.set_ydata(self.ts_c4_data)
            
            # Update y limits
            all_data = np.concatenate([self.ts_c3_data, self.ts_cz_data, self.ts_c4_data])
            if len(all_data) > 0:
                self.timeseries_ax.set_ylim(0, max(all_data) * 1.2 + 0.1)
            
            # Return updated artists
            return [self.radar_line, *self.bars, self.ts_c3_line, self.ts_cz_line, self.ts_c4_line]
        else:
            return []
    
    def update_topography(self, frame):
        """Update the brain topography visualizations"""
        if not self.use_real_data:
            # Different activity patterns based on classification state
            n = 20
            x = np.linspace(-1, 1, n)
            y = np.linspace(-1, 1, n)
            X, Y = np.meshgrid(x, y)
            
            # Create a circular mask
            mask = X**2 + Y**2 <= 1
            
            # Initialize data
            alpha_data = np.zeros((n, n))
            beta_data = np.zeros((n, n))
            
            # Add baseline activity
            alpha_data[mask] = np.random.normal(0.5, 0.1, np.sum(mask))
            beta_data[mask] = np.random.normal(0.2, 0.1, np.sum(mask))
            
            # Add state-specific patterns
            if self.current_classification == 'rest':
                # Rest: Strong alpha everywhere
                alpha_data[mask] = np.random.normal(0.8, 0.1, np.sum(mask))
                
            elif self.current_classification == 'left':
                # Left: ERD in right motor cortex (C4)
                # Create a right-sided focus
                right_mask = np.logical_and(mask, X > 0.2)
                alpha_data[right_mask] = np.random.normal(-0.5, 0.2, np.sum(right_mask))
                beta_data[right_mask] = np.random.normal(0.6, 0.2, np.sum(right_mask))
                
            elif self.current_classification == 'right':
                # Right: ERD in left motor cortex (C3)
                # Create a left-sided focus
                left_mask = np.logical_and(mask, X < -0.2)
                alpha_data[left_mask] = np.random.normal(-0.5, 0.2, np.sum(left_mask))
                beta_data[left_mask] = np.random.normal(0.6, 0.2, np.sum(left_mask))
            
            # Update the heatmaps
            self.alpha_heatmap.set_array(alpha_data)
            self.beta_heatmap.set_array(beta_data)
            
            return [self.alpha_heatmap, self.beta_heatmap]
        else:
            return []
    
    def update_classification(self):
        """Update the classification state"""
        if self.use_real_data and self.data_loaded and not self.paused:
            # Real data mode
            if self.epochs is not None and self.current_epoch < len(self.epochs):
                try:
                    # Get the event ID for this epoch
                    event_id = self.epochs.events[self.current_epoch, 2]
                    
                    # Convert to state name
                    event_dict = {v: k for k, v in self.epochs.event_id.items()}
                    if event_id in event_dict:
                        state = event_dict[event_id]
                        
                        # Map to our standard states if necessary
                        if state == 'hands':
                            state = 'left'
                        elif state == 'feet':
                            state = 'right'
                        
                        # Ensure it's one of our recognized states
                        if state not in class_names:
                            state = 'rest'
                            
                        self.update_state(state)
                except Exception as e:
                    print(f"Error determining state: {e}")
        
        elif not self.use_real_data:
            # Simulation mode
            if random.random() < 0.15:  # 15% chance to change state each time
                self.current_classification = random.choice(class_names)
                self.update_state(self.current_classification)
        
        # Schedule the next update
        self.root.after(2000, self.update_classification)
    
    def update_state(self, state):
        """Update the displayed state"""
        self.current_classification = state
        self.current_state.set(state.upper())
        
        # Update text color based on state
        if state == 'rest':
            self.state_value.configure(foreground=COLORS['states']['rest'])
        elif state == 'left':
            self.state_value.configure(foreground=COLORS['states']['left'])
        elif state == 'right':
            self.state_value.configure(foreground=COLORS['states']['right'])
        
        # Reset all boxes
        for state_key, (frame, label) in self.state_boxes.items():
            if state_key == state:
                # Active state
                label.configure(foreground='white')
                frame.configure(style='StateBox.TFrame')
                frame.configure(style='ActiveState.TFrame')
                # Update color based on state
                if state_key == 'rest':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['rest'])
                    self.state_value.configure(foreground=COLORS['states']['rest'])
                elif state_key == 'left':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['left'])
                    self.state_value.configure(foreground=COLORS['states']['left'])
                elif state_key == 'right':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['right'])
                    self.state_value.configure(foreground=COLORS['states']['right'])
            else:
                # Inactive state
                label.configure(foreground=COLORS['text'])
                frame.configure(style='StateBox.TFrame')
    
    def on_data_mode_change(self):
        """Handle change of data mode"""
        mode = self.data_mode_var.get()
        if mode == "Simulation":
            self.use_real_data = False
            self.subject_menu.config(state='disabled')
            self.task_menu.config(state='disabled')
            self.load_button.config(state='disabled')
            self.channel_selector.config(state='disabled')
            self.add_channel_button.config(state='disabled')
            self.clear_channels_button.config(state='disabled')
            self.prev_button.config(state='disabled')
            self.playpause_button.config(state='disabled')
            self.next_button.config(state='disabled')
            
            # Reset to default channels
            self.selected_channels = default_channels.copy()
            self.selected_channels_label.config(text=f"Selected: {', '.join([c.split(' ')[0] for c in self.selected_channels])}")
            
            # Reset titles
            for i, (ax, _) in enumerate(self.eeg_axes):
                if i < len(default_channels):
                    ax.set_title(default_channels[i], fontsize=12, color=COLORS['text'], fontweight='bold')
            
            self.status_var.set("Status: Using simulated data")
        else:
            self.use_real_data = True
            self.subject_menu.config(state='readonly')
            self.task_menu.config(state='readonly')
            self.load_button.config(state='normal')
            
            self.status_var.set("Status: Ready to load real data")
    
    def on_subject_change(self, event=None):
        """Handle subject change"""
        try:
            self.current_subject = int(self.subject_var.get())
        except ValueError:
            pass
    
    def on_task_change(self, event=None):
        """Handle task change"""
        self.current_task = self.task_var.get()
        
    def load_data(self):
        """Load real EEG data with progress updates"""
        self.status_var.set(f"Status: Loading data for Subject {self.current_subject}, {self.current_task}...")
        self.root.update()
        
        # Create progress bar
        progress_frame = ttk.Frame(self.data_selector_frame)
        progress_frame.grid(row=4, column=0, columnspan=5, padx=10, pady=5, sticky="ew")
        
        progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress_bar.pack(fill=tk.X, expand=True)
        progress_bar.start()
        
        # Reset data-related status
        self.data_loaded = False
        self.epochs = None
        self.raw_data = None
        self.selected_channels = []
        self.available_channels = []
        
        # Start a thread to load data
        self.thread = threading.Thread(target=self._load_data_thread)
        self.thread.daemon = True
        self.thread.start()
        
        # Monitor the thread
        self.check_loading_thread(progress_frame, progress_bar)
    
    def check_loading_thread(self, progress_frame, progress_bar):
        """Check if loading thread is complete"""
        if self.thread.is_alive():
            # Still loading, check again later
            self.root.after(100, lambda: self.check_loading_thread(progress_frame, progress_bar))
        else:
            # Thread completed, cleanup
            progress_bar.stop()
            progress_frame.destroy()
    
    def _load_data_thread(self):
        """Simplified data loading with minimal preprocessing"""
        try:
            # Get the selected task and subject
            runs = task_runs[self.current_task]
            subject_num = self.current_subject
            
            # Create simplified cache path
            cache_file = f"cache_S{subject_num}_{self.current_task}.pkl"
            
            # Try loading from cache first (using joblib which is faster than MNE for this)
            if os.path.exists(cache_file):
                self.root.after(0, lambda: self.status_var.set(f"Status: Loading cached data..."))
                # Load data using joblib instead of MNE's read_epochs
                cached_data = joblib.load(cache_file)
                self.epochs = cached_data['epochs']  
                self.channel_names = cached_data['channels']
                self.available_channels = cached_data['channels']
                self.time_axis = cached_data['times']
                self.fs = cached_data['fs']
                
                # Finish loading
                self.current_epoch = 0
                self.root.after(0, self.update_ui_after_loading)
                self.data_loaded = True
                return
            
            # Process from raw files - get minimal data needed
            raw_list = []
            
            # Download only if needed
            for run in runs:
                file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                                       f'S{str(subject_num).zfill(3)}', 
                                       f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
                
                if not os.path.exists(file_path):
                    # Only download the specific missing file
                    try:
                        self.root.after(0, lambda: self.status_var.set(f"Status: Downloading file for run {run}..."))
                        eegbci.load_data(subject_num, runs=[run], path='files/')
                    except Exception as e:
                        continue
                        
                # Load the file with minimal settings
                try:
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    raw_list.append(raw)
                except Exception as e:
                    continue
            
            if not raw_list:
                self.root.after(0, lambda: self.status_var.set("Status: No data found"))
                return
            
            # Very basic preprocessing - just extract events
            raw_concat = mne.concatenate_raws(raw_list)
            events, event_id = mne.events_from_annotations(raw_concat)
            
            # Create epochs with minimal settings
            event_id_selected = {k: v for k, v in event_id.items() if any(marker in k for marker in ['T0', 'T1', 'T2', 'T3', 'T4'])}
            epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                               tmin=0.5, tmax=3.5, baseline=None, preload=True)
            
            # Save to simple cache
            cache_data = {
                'epochs': epochs,
                'channels': epochs.ch_names,
                'times': epochs.times,
                'fs': int(epochs.info['sfreq'])
            }
            joblib.dump(cache_data, cache_file)
            
            # Store data in app
            self.epochs = epochs
            self.channel_names = epochs.ch_names
            self.available_channels = epochs.ch_names
            self.time_axis = epochs.times
            self.fs = int(epochs.info['sfreq'])
            
            # Set motor channels
            self.selected_channels = []
            for ch in ['C3', 'Cz', 'C4']:
                matching = [name for name in self.available_channels if ch in name]
                if matching:
                    self.selected_channels.append(matching[0])
            
            # If we don't have enough channels, use the first available ones
            while len(self.selected_channels) < 3 and len(self.available_channels) > len(self.selected_channels):
                next_ch = self.available_channels[len(self.selected_channels)]
                if next_ch not in self.selected_channels:
                    self.selected_channels.append(next_ch)
            
            # Update UI and finish
            self.current_epoch = 0
            self.root.after(0, self.update_ui_after_loading)
            self.data_loaded = True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Status: Error processing data: {e}"))
    
    def update_ui_after_loading(self):
        """Update UI after data is loaded"""
        # Enable channel selector
        self.channel_selector.config(state='readonly')
        self.channel_selector['values'] = self.available_channels
        self.add_channel_button.config(state='normal')
        self.clear_channels_button.config(state='normal')
        
        # Update channel display
        if self.selected_channels:
            self.selected_channels_label.config(text=f"Selected: {', '.join(self.selected_channels)}")
        
        # Enable navigation controls
        self.prev_button.config(state='normal')
        self.playpause_button.config(state='normal')
        self.next_button.config(state='normal')
        
        # Update epoch counter
        if self.epochs:
            self.epoch_var.set(f"Epoch: 1/{len(self.epochs)}")
        
        # Update status
        self.status_var.set(f"Status: Loaded data for Subject {self.current_subject}")
        
        # Indicate data is loaded
        self.data_loaded = True
    
    def add_channel(self):
        """Add a channel to the visualization"""
        channel = self.channel_selector.get()
        if channel and channel not in self.selected_channels:
            self.selected_channels.append(channel)
            self.selected_channels_label.config(text=f"Selected: {', '.join(self.selected_channels)}")
    
    def clear_channels(self):
        """Clear selected channels"""
        self.selected_channels = []
        self.selected_channels_label.config(text="Selected: None")
    
    def next_epoch(self):
        """Go to next epoch"""
        if self.epochs is not None and self.current_epoch < len(self.epochs) - 1:
            self.current_epoch += 1
            self.epoch_var.set(f"Epoch: {self.current_epoch + 1}/{len(self.epochs)}")
    
    def prev_epoch(self):
        """Go to previous epoch"""
        if self.epochs is not None and self.current_epoch > 0:
            self.current_epoch -= 1
            self.epoch_var.set(f"Epoch: {self.current_epoch + 1}/{len(self.epochs)}")
    
    def toggle_play_pause(self):
        """Toggle between play and pause states"""
        self.paused = not self.paused
        self.playpause_button.config(text="❚❚ Pause" if not self.paused else "► Play")

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGVisualization(root)
    root.mainloop()