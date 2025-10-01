#!/usr/bin/env python3
"""
Launcher script for 3D Object Recognition System
================================================

This script provides easy access to different components of the system.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_training():
    """Run the main training script"""
    print("🚀 Starting 3D Object Recognition Training...")
    subprocess.run([sys.executable, "modern_3d_recognition.py"])

def run_ui():
    """Run the Streamlit web UI"""
    print("🌐 Launching Web UI...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def run_tests():
    """Run the test suite"""
    print("🧪 Running Test Suite...")
    subprocess.run([sys.executable, "-m", "pytest", "test_system.py", "-v"])

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing Dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    parser = argparse.ArgumentParser(description="3D Object Recognition System Launcher")
    parser.add_argument("command", choices=["train", "ui", "test", "install"], 
                       help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_training()
    elif args.command == "ui":
        run_ui()
    elif args.command == "test":
        run_tests()
    elif args.command == "install":
        install_dependencies()

if __name__ == "__main__":
    main()
