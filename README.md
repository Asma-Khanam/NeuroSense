# NeuroSense

# Brain-Computer Interface (BCI) Motor Imagery Classification System

## Project Overview

This repository contains a comprehensive Brain-Computer Interface (BCI) system that processes electroencephalogram (EEG) signals to classify motor imagery commands and control a simulated prosthetic arm. The system translates brain activity patterns associated with imagined movements into control signals, enabling users to manipulate a virtual prosthetic through thought alone.

## Key Features

- **EEG Signal Processing**: Robust preprocessing pipeline for EEG signals including filtering, artifact removal, and feature extraction
- **Advanced Classification Models**: Implementation of multiple machine learning models (SVM, LDA, Gradient Boosting) with ensemble techniques for optimal performance
- **Interactive Visualizations**: Real-time visualization of EEG signals, feature extraction, and classification results
- **Prosthetic Arm Simulation**: A virtual prosthetic arm that responds to mental commands:
  - REST: Hold current position
  - LEFT: Open/close hand
  - RIGHT: Rotate wrist
- **Comprehensive UI**: User-friendly interfaces for monitoring BCI performance

## Technical Details

### Data Processing Pipeline

The system implements a sophisticated EEG processing pipeline that includes:
- Bandpass filtering (4-45 Hz) to isolate relevant neural oscillations
- Common Average Reference (CAR) spatial filtering
- ICA-based artifact removal for eye blinks and movements
- Motor cortex channel selection for improved signal quality

### Feature Extraction

Extracts a rich set of features from EEG signals:
- Spectral features (band powers in theta, alpha, beta ranges)
- Temporal features (variance, skewness, kurtosis)
- Connectivity features between key electrode pairs
- Specialized ERD/ERS (Event-Related Desynchronization/Synchronization) detection

### Classification

Multiple classification approaches with model selection:
- Support Vector Machines with optimized kernels
- Linear Discriminant Analysis with shrinkage
- Gradient Boosting classifiers
- Ensemble methods combining multiple models for enhanced accuracy

