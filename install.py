#!/usr/bin/env python
"""
Installation script for Multi-KPI Email Campaign Optimizer
Creates necessary directories and downloads required models
"""
import os
import sys
import subprocess
import argparse

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        'Data',
        'models',
        'Docs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def install_requirements():
    """Install Python dependencies"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def download_spacy_model():
    """Download Swedish SpaCy model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "sv_core_news_sm"])
        print("Successfully downloaded Swedish SpaCy model")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading SpaCy model: {e}")
        print("You can manually download it later with: python -m spacy download sv_core_news_sm")
        return False
    return True

def create_env_file():
    """Create a template .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Groq API Key for AI-powered subject line suggestions\n")
            f.write("GROQ_API_KEY=your-api-key-here\n")
        print("Created template .env file (please edit with your actual API key)")
    else:
        print(".env file already exists")

def create_sample_data():
    """Create sample data files if Data directory is empty"""
    if not os.path.exists('Data/example_delivery_data.csv'):
        with open('Data/example_delivery_data.csv', 'w') as f:
            f.write("InternalName;Subject;Preheader;Date;Sendouts;Opens;Clicks;Optouts;Dialog;Syfte;Product\n")
            f.write("DM123456;Take the car to your next adventure;Exclusive deals on car insurance;2024/06/10 15:59;14827;2559;211;9;F;VD;Mo\n")
            f.write("DM123457;Summer offers for you;Don't miss our summer deals;2024/06/15 10:30;12500;2100;180;7;P;VIN;BO_V_\n")
        print("Created example delivery data file")
    
    if not os.path.exists('Data/example_customer_data.csv'):
        with open('Data/example_customer_data.csv', 'w') as f:
            f.write("Primary key;OptOut;Open;Click;Gender;Age;InternalName;Bolag\n")
            f.write("12345678;0;1;0;Kvinna;69;DM123456;Stockholm\n")
            f.write("12345679;0;1;1;Man;45;DM123456;Stockholm\n")
            f.write("12345680;0;0;0;Kvinna;32;DM123457;Uppsala\n")
        print("Created example customer data file")

def main():
    parser = argparse.ArgumentParser(description="Install Multi-KPI Email Campaign Optimizer")
    parser.add_argument("--skip-requirements", action="store_true", help="Skip installing requirements")
    parser.add_argument("--skip-spacy", action="store_true", help="Skip downloading SpaCy model")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data files")
    
    args = parser.parse_args()
    
    print("Installing Multi-KPI Email Campaign Optimizer...")
    
    # Create necessary directories
    create_directories()
    
    # Install requirements
    if not args.skip_requirements:
        success = install_requirements()
        if not success:
            print("Warning: Failed to install requirements. You may need to manually install them.")
    
    # Download SpaCy model
    if not args.skip_spacy:
        success = download_spacy_model()
        if not success:
            print("Warning: Failed to download SpaCy model. You may need to manually download it.")
    
    # Create .env file template
    create_env_file()
    
    # Create sample data if requested
    if args.sample_data:
        create_sample_data()
    
    print("\nInstallation completed!")
    print("You can now run the application with: streamlit run app.py")

if __name__ == "__main__":
    main()