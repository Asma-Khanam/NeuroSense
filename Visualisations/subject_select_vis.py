import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import mne
from mne.datasets import eegbci

class SimplifiedEEGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Simplified EEG Viewer")
        self.root.geometry("800x600")
        
        # Controls frame
        controls_frame = ttk.Frame(root)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Subject selector
        ttk.Label(controls_frame, text="Subject:").grid(row=0, column=0, padx=5)
        self.subject_var = tk.StringVar(value="1")
        self.subject_menu = ttk.Combobox(controls_frame, textvariable=self.subject_var, 
                                         values=[str(i) for i in range(1, 11)], width=5)
        self.subject_menu.grid(row=0, column=1, padx=5)
        
        # Task selector
        ttk.Label(controls_frame, text="Task:").grid(row=0, column=2, padx=5)
        self.task_var = tk.StringVar(value="Task4")
        self.task_menu = ttk.Combobox(controls_frame, textvariable=self.task_var, 
                                      values=["Task4", "Task5"], width=8)
        self.task_menu.grid(row=0, column=3, padx=5)
        
        # Load button
        self.load_button = ttk.Button(controls_frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=0, column=4, padx=10)
        
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls_frame, textvariable=self.status_var).grid(row=0, column=5, padx=10)
        
        # Plot frame
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_data(self):
        """Load and display EEG data with minimal preprocessing"""
        self.status_var.set("Loading data...")
        self.root.update()
        
        try:
            # Get selected subject and task
            subject = int(self.subject_var.get())
            task = self.task_var.get()
            run = 4 if task == "Task4" else 8
            
            # Check if file exists
            file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                                   f'S{str(subject).zfill(3)}', 
                                   f'S{str(subject).zfill(3)}R{str(run).zfill(2)}.edf')
            
            if not os.path.exists(file_path):
                self.status_var.set("Downloading data...")
                self.root.update()
                eegbci.load_data(subject, runs=[run], path='files/')
            
            # Load raw data
            self.status_var.set("Processing data...")
            self.root.update()
            
            raw = mne.io.read_raw_edf(file_path, preload=True)
            
            # Skip heavy preprocessing, just get events
            events, event_id = mne.events_from_annotations(raw)
            
            # Create epochs with minimal processing
            epochs = mne.Epochs(raw, events, tmin=0.5, tmax=3.5, baseline=None, preload=True)
            
            # Plot first epoch
            self.fig.clear()
            
            # Get data for first epoch
            data = epochs.get_data()[0]
            times = epochs.times
            
            # Select a few key channels
            key_channels = ['C3..', 'Cz..', 'C4..']
            channel_indices = []
            
            for key in key_channels:
                for i, ch in enumerate(raw.ch_names):
                    if key in ch:
                        channel_indices.append(i)
                        break
            
            # If no key channels found, use first 3 channels
            if not channel_indices and len(raw.ch_names) >= 3:
                channel_indices = [0, 1, 2]
            
            # Plot channels
            for i, ch_idx in enumerate(channel_indices[:3]):
                ax = self.fig.add_subplot(3, 1, i+1)
                ax.plot(times, data[ch_idx])
                ax.set_title(raw.ch_names[ch_idx])
                ax.set_ylabel('µV')
                
                if i == 2:  # Add x-label on bottom plot
                    ax.set_xlabel('Time (s)')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.status_var.set(f"Loaded: Subject {subject}, {task}")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error loading data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimplifiedEEGViewer(root)
    root.mainloop()