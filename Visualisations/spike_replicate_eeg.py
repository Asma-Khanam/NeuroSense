import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import joblib
import time
import random
from scipy import signal
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Color scheme
COLORS = {
    'background': '#1e1e1e',       # Dark background
    'panel_bg': '#2d2d2d',         # Panel background
    'text': '#f0f0f0',             # Light text
    'signal': '#00ff00',           # Green signal
    'grid': '#444444',             # Grid color
    'buttons': '#555555',          # Button color
    'active_button': '#888888',    # Active button
    'states': {
        'rest': '#4CAF50',         # Green
        'left': '#2196F3',         # Blue
        'right': '#FF9800'         # Orange
    }
}

# Simulated EEG data parameters
SAMPLING_RATE = 250  # Hz
SIGNAL_LENGTH = 5    # seconds
BUFFER_SIZE = SAMPLING_RATE * SIGNAL_LENGTH
TIME_AXIS = np.linspace(0, SIGNAL_LENGTH, BUFFER_SIZE)

# Class names (these should match what your model was trained on)
CLASS_NAMES = ['rest', 'left', 'right']

class SpikeRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Spike Recorder")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS['background'])
        
        # Try to load the model
        try:
            self.model = joblib.load('high_acc_gb_model.pkl')
            print("Model loaded successfully!")
            self.have_model = True
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Running in simulation mode")
            self.have_model = False
        
        # Data buffers
        self.time_data = np.zeros(BUFFER_SIZE)
        self.signal_data = np.zeros((3, BUFFER_SIZE))  # 3 channels
        self.freq_data = np.zeros((40, BUFFER_SIZE // 10))  # Frequency data
        
        # Current state
        self.current_state = 'rest'
        self.paused = False
        
        # Setup UI
        self.setup_ui()
        
        # Start animations
        self.ani1 = animation.FuncAnimation(
            self.fig1, self.update_time_plot, 
            interval=100, blit=False
        )
        
        self.ani2 = animation.FuncAnimation(
            self.fig2, self.update_freq_plot, 
            interval=200, blit=False
        )
        
        # Start classification updates
        if not self.paused:
            self.update_classification()
    
    def setup_ui(self):
        """Set up the main user interface"""
        # Top control bar
        self.control_frame = tk.Frame(self.root, bg=COLORS['background'])
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Control buttons
        self.setup_buttons()
        
        # Main visualization area
        self.main_frame = tk.Frame(self.root, bg=COLORS['background'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Time domain plot (top)
        self.time_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        self.time_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=5)
        
        self.fig1 = Figure(figsize=(10, 4), dpi=100, facecolor=COLORS['background'])
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor(COLORS['background'])
        self.ax1.tick_params(colors=COLORS['text'])
        for spine in self.ax1.spines.values():
            spine.set_color(COLORS['grid'])
        
        self.line1, = self.ax1.plot(
            TIME_AXIS, 
            np.zeros(BUFFER_SIZE), 
            color=COLORS['signal'], 
            lw=1
        )
        
        self.ax1.set_ylim(-2, 2)
        self.ax1.set_xlim(0, SIGNAL_LENGTH)
        self.ax1.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        self.time_canvas = FigureCanvasTkAgg(self.fig1, self.time_frame)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frequency domain plot (bottom)
        self.freq_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        self.freq_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=5)
        
        self.fig2 = Figure(figsize=(10, 3), dpi=100, facecolor=COLORS['background'])
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor('blue')
        self.ax2.tick_params(colors=COLORS['text'])
        
        # Frequency domain visualization (spectrogram-like)
        self.spec_data = np.zeros((40, BUFFER_SIZE // 10))
        self.spectrum = self.ax2.imshow(
            self.spec_data,
            aspect='auto',
            cmap='viridis',
            origin='lower',
            vmin=0, vmax=1
        )
        
        # Add frequency labels (Hz)
        self.ax2.set_yticks([0, 10, 20, 30])
        self.ax2.set_yticklabels(['10 Hz', '20 Hz', '30 Hz', '40 Hz'])
        
        # Remove x axis ticks
        self.ax2.set_xticks([])
        
        self.freq_canvas = FigureCanvasTkAgg(self.fig2, self.freq_frame)
        self.freq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Playback controls at bottom
        self.playback_frame = tk.Frame(self.root, bg=COLORS['background'])
        self.playback_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add buttons with circular style
        button_size = 40
        
        # Rewind button
        self.rewind_button = tk.Canvas(
            self.playback_frame, width=button_size, height=button_size,
            bg=COLORS['background'], highlightthickness=0
        )
        self.rewind_button.pack(side=tk.LEFT, padx=5)
        self.rewind_button.create_oval(
            5, 5, button_size-5, button_size-5, 
            fill=COLORS['buttons'], outline=""
        )
        self.rewind_button.create_polygon(
            button_size//2+5, button_size//2, 
            button_size//2+10, button_size//2-7,
            button_size//2+10, button_size//2+7,
            fill="white"
        )
        self.rewind_button.create_polygon(
            button_size//2-5, button_size//2, 
            button_size//2, button_size//2-7,
            button_size//2, button_size//2+7,
            fill="white"
        )
        self.rewind_button.bind("<Button-1>", self.rewind)
        
        # Play/Pause button
        self.play_button = tk.Canvas(
            self.playback_frame, width=button_size, height=button_size,
            bg=COLORS['background'], highlightthickness=0
        )
        self.play_button.pack(side=tk.LEFT, padx=20)
        self.play_button.create_oval(
            5, 5, button_size-5, button_size-5, 
            fill=COLORS['buttons'], outline=""
        )
        self.play_pause_icon = self.play_button.create_rectangle(
            button_size//2-8, button_size//2-8,
            button_size//2-2, button_size//2+8,
            fill="white", outline=""
        )
        self.play_button.create_rectangle(
            button_size//2+2, button_size//2-8,
            button_size//2+8, button_size//2+8,
            fill="white", outline=""
        )
        self.play_button.bind("<Button-1>", self.toggle_play_pause)
        
        # Forward button
        self.forward_button = tk.Canvas(
            self.playback_frame, width=button_size, height=button_size,
            bg=COLORS['background'], highlightthickness=0
        )
        self.forward_button.pack(side=tk.LEFT, padx=5)
        self.forward_button.create_oval(
            5, 5, button_size-5, button_size-5, 
            fill=COLORS['buttons'], outline=""
        )
        self.forward_button.create_polygon(
            button_size//2-5, button_size//2, 
            button_size//2-10, button_size//2-7,
            button_size//2-10, button_size//2+7,
            fill="white"
        )
        self.forward_button.create_polygon(
            button_size//2+5, button_size//2, 
            button_size//2, button_size//2-7,
            button_size//2, button_size//2+7,
            fill="white"
        )
        self.forward_button.bind("<Button-1>", self.forward)
        
        # Status indicators on the right side
        self.status_frame = tk.Frame(self.playback_frame, bg=COLORS['background'])
        self.status_frame.pack(side=tk.RIGHT, padx=20)
        
        self.state_label = tk.Label(
            self.status_frame, 
            text="Current State: REST",
            font=("Arial", 12),
            fg=COLORS['states']['rest'],
            bg=COLORS['background']
        )
        self.state_label.pack(side=tk.RIGHT)
        
        # Add scale/time indicator
        self.scale_frame = tk.Frame(self.playback_frame, bg=COLORS['background'])
        self.scale_frame.pack(side=tk.RIGHT, padx=20)
        
        self.scale_canvas = tk.Canvas(
            self.scale_frame, width=100, height=20,
            bg=COLORS['background'], highlightthickness=0
        )
        self.scale_canvas.pack(side=tk.TOP)
        
        self.scale_canvas.create_line(
            0, 10, 80, 10, fill=COLORS['text'], width=1
        )
        self.scale_canvas.create_text(
            90, 10, text="1 s", fill=COLORS['text'], anchor="w"
        )
    
    def setup_buttons(self):
        """Set up the control buttons at the top"""
        button_frame = tk.Frame(self.control_frame, bg=COLORS['background'])
        button_frame.pack(side=tk.LEFT)
        
        # Button size
        btn_size = 60
        
        # Settings button
        self.settings_btn = tk.Canvas(
            button_frame, width=btn_size, height=btn_size,
            bg=COLORS['background'], highlightthickness=0
        )
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        self.settings_btn.create_oval(
            10, 10, btn_size-10, btn_size-10, 
            fill=COLORS['buttons'], outline=""
        )
        # Gear icon
        self.settings_btn.create_text(
            btn_size//2, btn_size//2, text="⚙️", font=("Arial", 20)
        )
        self.settings_btn.bind("<Button-1>", self.open_settings)
        
        # Wave button
        self.wave_btn = tk.Canvas(
            button_frame, width=btn_size, height=btn_size,
            bg=COLORS['background'], highlightthickness=0
        )
        self.wave_btn.pack(side=tk.LEFT, padx=5)
        self.wave_btn.create_oval(
            10, 10, btn_size-10, btn_size-10, 
            fill=COLORS['active_button'], outline=""
        )
        # Wave icon (sine wave)
        x = np.linspace(15, btn_size-15, 20)
        y = 5 * np.sin(x * 0.5) + btn_size//2
        for i in range(len(x)-1):
            self.wave_btn.create_line(
                x[i], y[i], x[i+1], y[i+1], 
                fill="white", width=2
            )
        self.wave_btn.bind("<Button-1>", self.toggle_wave_view)
        
        # FFT button
        self.fft_btn = tk.Canvas(
            button_frame, width=btn_size, height=btn_size,
            bg=COLORS['background'], highlightthickness=0
        )
        self.fft_btn.pack(side=tk.LEFT, padx=5)
        self.fft_btn.create_oval(
            10, 10, btn_size-10, btn_size-10, 
            fill=COLORS['buttons'], outline=""
        )
        # FFT text
        self.fft_btn.create_text(
            btn_size//2, btn_size//2, text="FFT", 
            font=("Arial", 14, "bold"), fill="white"
        )
        self.fft_btn.bind("<Button-1>", self.toggle_fft_view)
        
        # Current state indicator on the right
        self.model_label = tk.Label(
            self.control_frame, 
            text=f"Mode: {'Using Model' if self.have_model else 'Simulation'}",
            font=("Arial", 10),
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        self.model_label.pack(side=tk.RIGHT, padx=20)
    
    def generate_eeg_signal(self, state='rest', noise_level=0.5):
        """Generate simulated EEG for the given mental state"""
        # Base signal (alpha rhythm at 10Hz)
        base_signal = np.sin(2 * np.pi * 10 * TIME_AXIS)
        
        # Add noise
        noise = np.random.normal(0, noise_level, BUFFER_SIZE)
        
        # Different patterns for each state
        if state == 'rest':
            # REST: Strong alpha (10Hz) in all channels
            signal = base_signal * 1.5 + noise * 0.4
        elif state == 'left':
            # LEFT hand: Reduced alpha in right motor cortex
            signal = base_signal * 0.8 + noise * 0.7
            # Add some beta activity (15-20Hz)
            beta = np.sin(2 * np.pi * 18 * TIME_AXIS) * 0.6
            signal += beta
        elif state == 'right':
            # RIGHT hand: Reduced alpha in left motor cortex
            signal = base_signal * 0.7 + noise * 0.8
            # Add some beta activity (15-20Hz) with phase shift
            beta = np.sin(2 * np.pi * 18 * TIME_AXIS + 1) * 0.7
            signal += beta
        
        return signal
    
    def generate_spectrogram(self, state='rest'):
        """Generate simulated spectrogram data based on state"""
        spec_data = np.zeros((40, BUFFER_SIZE // 10))
        
        # Add background activity
        spec_data += np.random.rand(40, BUFFER_SIZE // 10) * 0.1
        
        # Add state-specific patterns
        if state == 'rest':
            # Strong alpha (8-12 Hz)
            alpha_band = slice(8, 12)
            spec_data[alpha_band, :] += 0.8 + np.random.rand(4, BUFFER_SIZE // 10) * 0.2
            
        elif state == 'left':
            # Reduced alpha, increased beta (15-25 Hz)
            alpha_band = slice(8, 12)
            beta_band = slice(15, 25)
            
            spec_data[alpha_band, :] += 0.3 + np.random.rand(4, BUFFER_SIZE // 10) * 0.1
            spec_data[beta_band, :] += 0.7 + np.random.rand(10, BUFFER_SIZE // 10) * 0.3
            
        elif state == 'right':
            # Reduced alpha, different beta pattern
            alpha_band = slice(8, 12)
            beta_band = slice(15, 25)
            
            spec_data[alpha_band, :] += 0.3 + np.random.rand(4, BUFFER_SIZE // 10) * 0.1
            
            # Create a patterned beta activity
            for i in range(beta_band.start, beta_band.stop):
                factor = 0.3 + 0.7 * np.sin(i * 0.5)
                spec_data[i, :] += factor * (0.5 + np.random.rand(BUFFER_SIZE // 10) * 0.5)
        
        return spec_data
    
    def update_time_plot(self, frame):
        """Update the time domain plot"""
        # Generate new signal data based on current state
        signal = self.generate_eeg_signal(self.current_state)
        
        # Update plot
        self.line1.set_ydata(signal)
        
        return [self.line1]
    
    def update_freq_plot(self, frame):
        """Update the frequency domain visualization"""
        # Generate or update spectrogram data
        self.spec_data = self.generate_spectrogram(self.current_state)
        
        # Update the visualization
        self.spectrum.set_array(self.spec_data)
        
        return [self.spectrum]
    
    def update_classification(self):
        """Update the classification state"""
        if not self.paused:
            # In a real application, this would get data from the model
            if self.have_model:
                # Simulate feature extraction and model prediction
                # In reality, this would process actual EEG data
                features = np.random.random((1, 84))  # Adjust based on your feature dimensions
                
                try:
                    prediction = self.model.predict(features)
                    predicted_class = CLASS_NAMES[prediction[0] % len(CLASS_NAMES)]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_class = random.choice(CLASS_NAMES)
            else:
                # Simple simulation mode
                predicted_class = random.choice(CLASS_NAMES)
            
            # Update state
            self.update_state(predicted_class)
        
        # Schedule the next update
        self.root.after(2000, self.update_classification)
    
    def update_state(self, state):
        """Update the displayed state"""
        self.current_state = state
        
        # Update the state label
        self.state_label.config(
            text=f"Current State: {state.upper()}",
            fg=COLORS['states'][state]
        )
    
    def toggle_play_pause(self, event=None):
        """Toggle between play and pause states"""
        self.paused = not self.paused
        
        if self.paused:
            # Update the play button to show play icon
            self.play_button.delete(self.play_pause_icon)
            self.play_pause_icon = self.play_button.create_polygon(
                20, 13, 20, 27, 32, 20, 
                fill="white", outline=""
            )
        else:
            # Update the play button to show pause icon
            self.play_button.delete(self.play_pause_icon)
            self.play_pause_icon = self.play_button.create_rectangle(
                18, 13, 22, 27,
                fill="white", outline=""
            )
            self.play_button.create_rectangle(
                26, 13, 30, 27,
                fill="white", outline=""
            )
            
            # Resume updates
            self.update_classification()
    
    def rewind(self, event=None):
        """Rewind functionality"""
        # In a real application, this would rewind the data
        pass
    
    def forward(self, event=None):
        """Forward functionality"""
        # In a real application, this would fast-forward the data
        pass
    
    def toggle_wave_view(self, event=None):
        """Toggle between different wave views"""
        # Highlight wave button, un-highlight FFT button
        self.wave_btn.create_oval(
            10, 10, 50, 50, 
            fill=COLORS['active_button'], outline=""
        )
        self.fft_btn.create_oval(
            10, 10, 50, 50, 
            fill=COLORS['buttons'], outline=""
        )
        
        # Redraw button contents
        x = np.linspace(15, 45, 20)
        y = 5 * np.sin(x * 0.5) + 30
        for i in range(len(x)-1):
            self.wave_btn.create_line(
                x[i], y[i], x[i+1], y[i+1], 
                fill="white", width=2
            )
        
        self.fft_btn.create_text(
            30, 30, text="FFT", 
            font=("Arial", 14, "bold"), fill="white"
        )
    
    def toggle_fft_view(self, event=None):
        """Toggle to FFT view"""
        # Highlight FFT button, un-highlight wave button
        self.fft_btn.create_oval(
            10, 10, 50, 50, 
            fill=COLORS['active_button'], outline=""
        )
        self.wave_btn.create_oval(
            10, 10, 50, 50, 
            fill=COLORS['buttons'], outline=""
        )
        
        # Redraw button contents
        self.fft_btn.create_text(
            30, 30, text="FFT", 
            font=("Arial", 14, "bold"), fill="white"
        )
        
        x = np.linspace(15, 45, 20)
        y = 5 * np.sin(x * 0.5) + 30
        for i in range(len(x)-1):
            self.wave_btn.create_line(
                x[i], y[i], x[i+1], y[i+1], 
                fill="white", width=2
            )
    
    def open_settings(self, event=None):
        """Open settings menu"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg=COLORS['background'])
        
        # Add settings options
        settings_label = tk.Label(
            settings_window,
            text="EEG Spike Recorder Settings",
            font=("Arial", 14, "bold"),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        settings_label.pack(pady=20)
        
        # Example settings
        # Sampling rate
        rate_frame = tk.Frame(settings_window, bg=COLORS['background'])
        rate_frame.pack(fill=tk.X, padx=20, pady=5)
        
        rate_label = tk.Label(
            rate_frame, 
            text="Sampling Rate:", 
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        rate_label.pack(side=tk.LEFT)
        
        rate_var = tk.StringVar(value="250 Hz")
        rate_menu = ttk.Combobox(
            rate_frame, 
            textvariable=rate_var,
            values=["125 Hz", "250 Hz", "500 Hz", "1000 Hz"],
            width=10
        )
        rate_menu.pack(side=tk.RIGHT)
        
        # Noise level
        noise_frame = tk.Frame(settings_window, bg=COLORS['background'])
        noise_frame.pack(fill=tk.X, padx=20, pady=5)
        
        noise_label = tk.Label(
            noise_frame, 
            text="Noise Level:", 
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        noise_label.pack(side=tk.LEFT)
        
        noise_scale = ttk.Scale(
            noise_frame, 
            from_=0.1, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            value=0.5,
            length=200
        )
        noise_scale.pack(side=tk.RIGHT)
        
        # Model selection
        model_frame = tk.Frame(settings_window, bg=COLORS['background'])
        model_frame.pack(fill=tk.X, padx=20, pady=5)
        
        model_label = tk.Label(
            model_frame, 
            text="Model File:", 
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        model_label.pack(side=tk.LEFT)
        
        model_entry = ttk.Entry(model_frame, width=20)
        model_entry.insert(0, "high_acc_gb_model.pkl")
        model_entry.pack(side=tk.LEFT, padx=5)
        
        browse_btn = ttk.Button(model_frame, text="Browse")
        browse_btn.pack(side=tk.RIGHT)
        
        # Close button
        close_btn = ttk.Button(
            settings_window, 
            text="Apply and Close",
            command=settings_window.destroy
        )
        close_btn.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpikeRecorderApp(root)
    root.mainloop()