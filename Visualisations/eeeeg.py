import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import os
import mne
from mne.datasets import eegbci
import time
import threading
import joblib
import random

# Color scheme
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
}

# Buffer size for visualization
BUFFER_SIZE = 1000

# Class names
CLASS_NAMES = ['rest', 'left', 'right']

# Define runs for task 4 and task 5
TASK_RUNS = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

class RealEEGVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("Real EEG Data Visualization")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['background'])
        
        # Try to load the model
        try:
            self.model = joblib.load('models/high_acc_gb_model.pkl')
            print("Model loaded successfully!")
            self.have_model = True
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Running in simulation mode")
            self.have_model = False
        
        # Dataset info
        self.subjects = [i for i in range(1, 11)]  # Subject IDs from 1-10
        self.current_subject = 1
        self.current_task = 'Task4'
        self.current_run = 4
        self.current_epoch = 0
        
        # EEG data
        self.epochs = None
        self.raw_data = None
        self.channel_names = None
        self.fs = 160  # Default sampling rate
        self.time_axis = None
        self.paused = False
        self.data_loaded = False
        
        # Current state
        self.current_state = 'rest'
        
        # Setup UI
        self.setup_ui()
        
        # Load initial data
        self.load_subject_data(1)
        
        # Start animations
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, 
            interval=100, blit=False
        )
        
        # Start classification updates
        if not self.paused:
            self.update_classification()
    
    def setup_ui(self):
        """Set up the main user interface"""
        # Top control bar
        self.control_frame = tk.Frame(self.root, bg=COLORS['background'])
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Subject selector
        self.subject_label = tk.Label(
            self.control_frame,
            text="Subject:",
            font=("Arial", 12),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        self.subject_label.pack(side=tk.LEFT, padx=5)
        
        self.subject_var = tk.StringVar(value="1")
        self.subject_menu = ttk.Combobox(
            self.control_frame,
            textvariable=self.subject_var,
            values=[str(i) for i in self.subjects],
            width=5
        )
        self.subject_menu.pack(side=tk.LEFT, padx=5)
        self.subject_menu.bind("<<ComboboxSelected>>", self.on_subject_change)
        
        # Task selector
        self.task_label = tk.Label(
            self.control_frame,
            text="Task:",
            font=("Arial", 12),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        self.task_label.pack(side=tk.LEFT, padx=5)
        
        self.task_var = tk.StringVar(value="Task4")
        self.task_menu = ttk.Combobox(
            self.control_frame,
            textvariable=self.task_var,
            values=list(TASK_RUNS.keys()),
            width=8
        )
        self.task_menu.pack(side=tk.LEFT, padx=5)
        self.task_menu.bind("<<ComboboxSelected>>", self.on_task_change)
        
        # Load button
        self.load_button = ttk.Button(
            self.control_frame,
            text="Load Data",
            command=self.on_load_data
        )
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        # Status indicator on the right
        self.status_label = tk.Label(
            self.control_frame,
            text="Status: Not loaded",
            font=("Arial", 10),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        # Main visualization area
        self.main_frame = tk.Frame(self.root, bg=COLORS['background'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for multiple visualizations
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # EEG Signal Tab
        self.eeg_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eeg_tab, text="EEG Signals")
        
        # Create EEG plot
        self.fig = Figure(figsize=(10, 6), dpi=100)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.eeg_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # We'll create axes dynamically once we know the channel count
        self.axes = []
        self.lines = []
        
        # Control panel at the bottom
        self.bottom_frame = tk.Frame(self.main_frame, bg=COLORS['background'])
        self.bottom_frame.pack(fill=tk.X, pady=10)
        
        # State boxes
        self.states_frame = tk.Frame(self.bottom_frame, bg=COLORS['background'])
        self.states_frame.pack(pady=10)
        
        self.state_boxes = {}
        
        for i, state in enumerate(CLASS_NAMES):
            state_frame = tk.Frame(
                self.states_frame, 
                width=100, 
                height=60, 
                bg='gray',
                relief=tk.RAISED,
                bd=2
            )
            state_frame.pack(side=tk.LEFT, padx=15)
            state_frame.pack_propagate(False)  # Prevent the frame from shrinking
            
            state_label = tk.Label(
                state_frame,
                text=state.upper(),
                font=("Arial", 14, "bold"),
                bg='gray',
                fg='white'
            )
            state_label.pack(expand=True, fill=tk.BOTH)
            
            self.state_boxes[state] = (state_frame, state_label)
        
        # Play/Pause and Epoch Selector
        self.playback_frame = tk.Frame(self.bottom_frame, bg=COLORS['background'])
        self.playback_frame.pack(pady=10)
        
        self.prev_button = ttk.Button(
            self.playback_frame,
            text="◄ Prev",
            command=self.prev_epoch
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.playpause_button = ttk.Button(
            self.playback_frame,
            text="❚❚ Pause" if not self.paused else "► Play",
            command=self.toggle_play_pause
        )
        self.playpause_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(
            self.playback_frame,
            text="Next ►",
            command=self.next_epoch
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.epoch_label = tk.Label(
            self.playback_frame,
            text="Epoch: 0/0",
            font=("Arial", 10),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        self.epoch_label.pack(side=tk.LEFT, padx=20)
        
        # Current state display
        self.state_frame = tk.Frame(self.bottom_frame, bg=COLORS['background'])
        self.state_frame.pack(pady=5)
        
        self.state_label = tk.Label(
            self.state_frame,
            text="Detected Mental Command:",
            font=("Arial", 14),
            bg=COLORS['background'],
            fg=COLORS['text']
        )
        self.state_label.pack(side=tk.LEFT, padx=10)
        
        self.current_state_var = tk.StringVar()
        self.current_state_var.set("INITIALIZING...")
        
        self.state_value = tk.Label(
            self.state_frame,
            textvariable=self.current_state_var,
            font=("Arial", 16, "bold"),
            bg=COLORS['background'],
            fg=COLORS['secondary']
        )
        self.state_value.pack(side=tk.LEFT, padx=10)
    
    def load_subject_data(self, subject_num):
        """Load actual EEG data for a specific subject"""
        self.status_label.config(text="Status: Loading data...")
        self.root.update()
        
        # Create a loading thread to avoid UI freeze
        thread = threading.Thread(target=self._load_data_thread, args=(subject_num,))
        thread.daemon = True
        thread.start()
    
    def _load_data_thread(self, subject_num):
        """Background thread for data loading"""
        try:
            # Reset data
            self.data_loaded = False
            self.epochs = None
            self.raw_data = None
            
            # Get the selected task
            task = self.current_task
            runs = TASK_RUNS[task]
            
            raw_list = []
            
            # Create proper file paths
            for run in runs:
                file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                                       f'S{str(subject_num).zfill(3)}', 
                                       f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
                
                try:
                    print(f"Attempting to load: {file_path}")
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                    raw_list.append(raw)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    # Try to download
                    try:
                        print(f"Attempting to download run {run} for subject {subject_num}")
                        eegbci.load_data(subject_num, runs=[run], path='files/')
                        raw = mne.io.read_raw_edf(file_path, preload=True)
                        print(f"Successfully downloaded and loaded data")
                        raw_list.append(raw)
                    except Exception as e2:
                        print(f"Download failed: {e2}")
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Could not download data: {e2}"))
                        self.root.after(0, lambda: self.status_label.config(text="Status: Error loading data"))
                        return
            
            if not raw_list:
                self.root.after(0, lambda: messagebox.showerror("Error", "No data loaded for subject. Please check if files exist."))
                self.root.after(0, lambda: self.status_label.config(text="Status: No data found"))
                return
            
            # Concatenate runs
            raw_concat = mne.concatenate_raws(raw_list)
            print(f"Concatenated raw data with {raw_concat.n_times} total samples")
            
            # Apply preprocessing (simplified version of what's in train2.py)
            raw_concat.filter(l_freq=1.0, h_freq=None)
            raw_concat.notch_filter(freqs=[50, 60])
            raw_concat.set_eeg_reference('average', projection=False)
            raw_concat.filter(l_freq=4, h_freq=45)
            
            # Select motor cortex channels if possible
            motor_channels = ['C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 
                           'FC3.', 'FC1.', 'FCz.', 'FC2.', 'FC4.', 
                           'CP3.', 'CP1.', 'CPz.', 'CP2.', 'CP4.',
                           'P3..', 'Pz..', 'P4..']
            
            # Get available motor channels
            available_motor = [ch for ch in motor_channels if ch in raw_concat.ch_names]
            
            if available_motor and len(available_motor) >= 3:
                print(f"Selecting {len(available_motor)} motor-related channels")
                raw_concat.pick_channels(available_motor)
            
            # Extract events
            events, event_id = mne.events_from_annotations(raw_concat)
            print(f"Found events: {event_id}")
            
            # Define events of interest
            event_id_selected = {
                'rest': event_id.get('T0', event_id.get('rest', None)),
                'left': event_id.get('T1', event_id.get('left', None)),
                'right': event_id.get('T2', event_id.get('right', None)),
                'hands': event_id.get('T3', event_id.get('hands', None)),
                'feet': event_id.get('T4', event_id.get('feet', None))
            }
            event_id_selected = {k: v for k, v in event_id_selected.items() if v is not None}
            
            if not event_id_selected:
                self.root.after(0, lambda: messagebox.showerror("Error", "No relevant events found for subject."))
                self.root.after(0, lambda: self.status_label.config(text="Status: No events found"))
                return
            
            # Create epochs
            epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                               tmin=0.5, tmax=3.5, baseline=(0.5, 1.0), preload=True)
            
            print(f"Created {len(epochs)} epochs with shape: {epochs.get_data().shape}")
            
            # Store data
            self.raw_data = raw_concat
            self.epochs = epochs
            self.channel_names = raw_concat.ch_names
            self.fs = int(raw_concat.info['sfreq'])
            self.current_epoch = 0
            
            # Create time axis based on sampling rate
            epoch_length = epochs.times.shape[0]
            self.time_axis = epochs.times
            
            # Update UI from main thread
            self.root.after(0, self.setup_plot_axes)
            self.root.after(0, lambda: self.status_label.config(text=f"Status: Loaded subject {subject_num}, {len(epochs)} epochs"))
            self.root.after(0, lambda: self.epoch_label.config(text=f"Epoch: 1/{len(epochs)}"))
            
            # Set data loaded flag
            self.data_loaded = True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing data: {e}"))
            self.root.after(0, lambda: self.status_label.config(text="Status: Error processing data"))
    
    def setup_plot_axes(self):
        """Set up plotting axes based on loaded channels"""
        if self.epochs is None or len(self.channel_names) == 0:
            return
        
        # Clear existing axes and lines
        self.fig.clear()
        self.axes = []
        self.lines = []
        
        # Create subplot for each channel
        for i, ch_name in enumerate(self.channel_names):
            ax = self.fig.add_subplot(len(self.channel_names), 1, i+1)
            ax.set_title(ch_name, fontsize=10)
            ax.set_ylabel('μV', fontsize=8)
            
            if i == len(self.channel_names) - 1:  # Add x-label only for bottom plot
                ax.set_xlabel('Time (s)', fontsize=10)
            
            # Create line with empty data
            line, = ax.plot(self.time_axis, np.zeros_like(self.time_axis))
            
            # Add to lists
            self.axes.append(ax)
            self.lines.append(line)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_plot(self, frame):
        """Update the plot with actual EEG data"""
        if not self.data_loaded or self.epochs is None or self.paused:
            return self.lines
        
        # Get data for current epoch
        epoch_data = self.epochs.get_data()[self.current_epoch]
        
        # Update each line
        for i, line in enumerate(self.lines):
            if i < epoch_data.shape[0]:  # Ensure we don't try to access non-existent channels
                line.set_ydata(epoch_data[i])
                
                # Update axis limits with padding
                self.axes[i].set_ylim(np.min(epoch_data[i]) - 1, np.max(epoch_data[i]) + 1)
        
        # Try to determine the state from epoch metadata
        try:
            event_id = self.epochs.events[self.current_epoch, 2]
            reverse_event_id = {v: k for k, v in self.epochs.event_id.items()}
            if event_id in reverse_event_id:
                self.current_state = reverse_event_id[event_id]
                self.update_state(self.current_state)
        except Exception as e:
            print(f"Error determining state: {e}")
        
        return self.lines
    
    def update_classification(self):
        """Update the classification state"""
        if not self.paused and self.data_loaded and self.epochs is not None:
            if self.have_model:
                # Get features for current epoch
                try:
                    # In a real implementation, you would use the actual feature extraction
                    # function from train2.py to extract features from the current epoch
                    # This is simplified and will not work - just for demonstration
                    
                    # Use the actual epoch class for now
                    event_id = self.epochs.events[self.current_epoch, 2]
                    reverse_event_id = {v: k for k, v in self.epochs.event_id.items()}
                    if event_id in reverse_event_id:
                        predicted_class = reverse_event_id[event_id]
                    else:
                        predicted_class = 'rest'  # Default
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_class = 'rest'
            else:
                # Just use the epoch's actual class
                try:
                    event_id = self.epochs.events[self.current_epoch, 2]
                    reverse_event_id = {v: k for k, v in self.epochs.event_id.items()}
                    if event_id in reverse_event_id:
                        predicted_class = reverse_event_id[event_id]
                    else:
                        predicted_class = 'rest'  # Default
                except:
                    predicted_class = 'rest'
            
            # Update state
            self.update_state(predicted_class)
            
            # Move to next epoch every 3 seconds if not paused
            if not self.paused:
                self.root.after(3000, self.next_epoch)
        
        # Schedule the next update
        self.root.after(1000, self.update_classification)
    
    def update_state(self, state):
        """Update the displayed state"""
        # Normalize state name to match expected format
        if state in CLASS_NAMES:
            self.current_state = state
        elif state == 'hands':
            self.current_state = 'left'  # Map 'hands' to 'left' for visualization
        elif state == 'feet':
            self.current_state = 'right'  # Map 'feet' to 'right' for visualization
        else:
            self.current_state = 'rest'  # Default
        
        # Update state display
        self.current_state_var.set(self.current_state.upper())
        
        # Update state color
        if self.current_state == 'rest':
            color = COLORS['states']['rest']
        elif self.current_state == 'left':
            color = COLORS['states']['left']
        elif self.current_state == 'right':
            color = COLORS['states']['right']
        else:
            color = 'gray'
        
        self.state_value.config(fg=color)
        
        # Reset all boxes to gray
        for state_key, (frame, label) in self.state_boxes.items():
            frame.config(bg='gray')
            label.config(bg='gray')
        
        # Highlight the active state
        if self.current_state in self.state_boxes:
            frame, label = self.state_boxes[self.current_state]
            frame.config(bg=color)
            label.config(bg=color)
    
    def next_epoch(self):
        """Go to next epoch"""
        if self.epochs is not None and self.current_epoch < len(self.epochs) - 1:
            self.current_epoch += 1
            self.epoch_label.config(text=f"Epoch: {self.current_epoch+1}/{len(self.epochs)}")
    
    def prev_epoch(self):
        """Go to previous epoch"""
        if self.epochs is not None and self.current_epoch > 0:
            self.current_epoch -= 1
            self.epoch_label.config(text=f"Epoch: {self.current_epoch+1}/{len(self.epochs)}")
    
    def toggle_play_pause(self):
        """Toggle between play and pause states"""
        self.paused = not self.paused
        self.playpause_button.config(text="❚❚ Pause" if not self.paused else "► Play")
    
    def on_subject_change(self, event=None):
        """Handle subject change"""
        try:
            subject = int(self.subject_var.get())
            self.current_subject = subject
        except ValueError:
            pass
    
    def on_task_change(self, event=None):
        """Handle task change"""
        task = self.task_var.get()
        if task in TASK_RUNS:
            self.current_task = task
    
    def on_load_data(self):
        """Handle the Load Data button click"""
        self.load_subject_data(self.current_subject)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealEEGVisualization(root)
    root.mainloop()