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
    model = joblib.load('high_acc_gb_model.pkl')  # You can change this to your preferred model
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
channels = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']

# Class names
class_names = ['rest', 'left', 'right']

# Signal patterns for different states
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
        
        # State boxes (using a more modern approach)
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
        
        # Start the animation
        self.ani = animation.FuncAnimation(
            self.eeg_fig, self.update_eeg_plot, 
            interval=100,  # update every 100ms
            blit=False
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
            ax.set_title(channels[i], fontsize=12, color=COLORS['text'], fontweight='bold')
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
        
        # Add an explanation panel
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
        
        # Add feature animation
        self.feature_ani = animation.FuncAnimation(
            self.feature_fig, self.update_feature_plots, 
            interval=200,  # update every 200ms
            blit=False
        )
    
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
        
        # Add animation for the topography
        self.topo_ani = animation.FuncAnimation(
            self.topo_fig, self.update_topography, 
            interval=200,  # update every 200ms
            blit=False
        )
    
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
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Activity', fontsize=8, color=COLORS['text'])
        
        return im
    
    def update_eeg_plot(self, frame):
        """Update the EEG plot"""
        # Get new data based on current classification
        self.current_signals = generate_eeg_signal(self.current_classification)
        
        # Generate a simulated spectrogram
        spec_data = np.zeros((30, 100))
        
        # Add alpha band (8-13 Hz) activity
        alpha_idx = slice(8, 13)
        
        if self.current_classification == 'rest':
            # Strong alpha in all channels
            spec_data[alpha_idx, :] = 0.8 + np.random.rand(5, 100) * 0.2
            highlight_colors = [COLORS['states']['rest']] * 3
            highlight_opacity = [0.2] * 3
            annotation = "Strong regular alpha waves visible in ALL channels - typical REST pattern"
            
        elif self.current_classification == 'left':
            # Reduced alpha in right motor cortex (C4)
            spec_data[alpha_idx, :] = 0.8 + np.random.rand(5, 100) * 0.2
            # Add ERD in the middle portion for C4
            spec_data[alpha_idx, 40:70] = 0.3 + np.random.rand(5, 30) * 0.2
            
            highlight_colors = [COLORS['states']['left']] * 3
            highlight_opacity = [0.1, 0.1, 0.4]  # Highlight C4 (right motor cortex) more strongly
            annotation = "Reduced alpha waves in C4 (right motor cortex) - LEFT hand movement pattern"
            
        elif self.current_classification == 'right':
            # Reduced alpha in left motor cortex (C3)
            spec_data[alpha_idx, :] = 0.8 + np.random.rand(5, 100) * 0.2
            # Add ERD in the middle portion for C3
            spec_data[alpha_idx, 40:70] = 0.3 + np.random.rand(5, 30) * 0.2
            
            highlight_colors = [COLORS['states']['right']] * 3
            highlight_opacity = [0.4, 0.1, 0.1]  # Highlight C3 (left motor cortex) more strongly
            annotation = "Reduced alpha waves in C3 (left motor cortex) - RIGHT hand movement pattern"
        
        # Add some beta activity (13-30 Hz)
        beta_idx = slice(13, 30)
        spec_data[beta_idx, :] = 0.3 + np.random.rand(beta_idx.stop - beta_idx.start, 100) * 0.2
        
        # Add some theta activity (4-8 Hz)
        theta_idx = slice(4, 8)
        spec_data[theta_idx, :] = 0.5 + np.random.rand(4, 100) * 0.2
        
        # Update spectrogram data
        self.spectrogram.set_array(spec_data)
        
        self.status_text.set(annotation)
        
        # Update each channel plot
        for i, (ax, line) in enumerate(self.eeg_axes):
            # Update the line data
            line.set_ydata(self.current_signals[i])
            
            # Update y-axis limits with some padding
            ax.set_ylim(min(self.current_signals[i]) - 0.5, max(self.current_signals[i]) + 0.5)
            
            # Get the y-axis limits for highlighting
            ylim = ax.get_ylim()
            height = ylim[1] - ylim[0]
            
            # Remove old highlight
            self.highlight_boxes[i].remove()
            
            # Make all ERD annotations invisible initially
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
        
        # Return the updated lines
        return [line for _, line in self.eeg_axes]
    
    def update_feature_plots(self, frame):
        """Update the feature visualizations"""
        # Extract features from current signals
        features = extract_simple_features(self.current_signals)
        
        # Update radar chart data
        # First 9 features are the band powers (3 channels x 3 frequency bands)
        radar_data = features[0, :9]
        
        # Update radar chart
        self.radar_values = radar_data.tolist()
        self.radar_values += self.radar_values[:1]  # Close the loop
        self.radar_line.set_ydata(self.radar_values)
        
        # Update the fill
        self.radar_ax.collections[0].remove()
        self.radar_ax.fill(self.radar_angles, self.radar_values, 
                         color=COLORS['primary'], alpha=0.25)
        
        # Update bar chart
        for i, bar in enumerate(self.bars):
            bar.set_height(radar_data[i])
        
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
    
    def update_topography(self, frame):
        """Update the brain topography visualizations"""
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
        
        # Reset all boxes
        for s, (frame, label) in self.state_boxes.items():
            if s == state:
                # Active state
                label.configure(foreground='white')
                frame.configure(style='StateBox.TFrame')
                frame.configure(style='ActiveState.TFrame')
                # Update color based on state
                if s == 'rest':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['rest'])
                    self.state_value.configure(foreground=COLORS['states']['rest'])
                elif s == 'left':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['left'])
                    self.state_value.configure(foreground=COLORS['states']['left'])
                elif s == 'right':
                    self.style.configure('ActiveState.TFrame', background=COLORS['states']['right'])
                    self.state_value.configure(foreground=COLORS['states']['right'])
            else:
                # Inactive state
                label.configure(foreground=COLORS['text'])
                frame.configure(style='StateBox.TFrame')

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGVisualization(root)
    root.mainloop()