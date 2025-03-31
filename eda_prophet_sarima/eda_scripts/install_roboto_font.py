#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to download and install the Roboto font for use with matplotlib
"""

import os
import sys
import subprocess
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import shutil
from pathlib import Path

def is_font_installed():
    """Check if Roboto font is already installed and available to matplotlib"""
    fonts = [f.name for f in fm.fontManager.ttflist]
    return any('Roboto' in font for font in fonts)

def install_roboto_font():
    """Download and install Roboto font for matplotlib"""
    print("Installing Roboto font for matplotlib...")
    
    # Create temporary directory
    temp_dir = Path('temp_fonts')
    temp_dir.mkdir(exist_ok=True)
    
    # Define Roboto font variants to download (using Google Fonts API)
    roboto_variants = {
        "Roboto-Regular.ttf": "https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxP.ttf",
        "Roboto-Bold.ttf": "https://fonts.gstatic.com/s/roboto/v30/KFOlCnqEu92Fr1MmWUlfBBc9.ttf",
        "Roboto-Italic.ttf": "https://fonts.gstatic.com/s/roboto/v30/KFOkCnqEu92Fr1Mu51xIIzc.ttf",
        "Roboto-BoldItalic.ttf": "https://fonts.gstatic.com/s/roboto/v30/KFOjCnqEu92Fr1Mu51TzBic6CsE.ttf"
    }
    
    try:
        # Get matplotlib font directory using font_manager module
        fonts_dir = Path(fm.findfont(fm.FontProperties(family='DejaVu Sans'))).parent
        
        print(f"Font installation directory: {fonts_dir}")
        
        # Download and install each variant
        installed_count = 0
        for font_name, font_url in roboto_variants.items():
            font_path = temp_dir / font_name
            
            print(f"Downloading {font_name}...")
            try:
                urllib.request.urlretrieve(font_url, font_path)
                
                # Copy font file to matplotlib fonts directory
                dest_path = fonts_dir / font_name
                shutil.copy(font_path, dest_path)
                print(f"Installed {font_name}")
                installed_count += 1
            except Exception as e:
                print(f"Failed to download {font_name}: {e}")
        
        if installed_count > 0:
            # Update matplotlib font cache by recreating the font manager
            print("Updating matplotlib font cache...")
            # Cache invalidation - let matplotlib rebuild its cache on next access
            fm.fontManager = fm.FontManager()
            
            print(f"Successfully installed {installed_count}/{len(roboto_variants)} Roboto font variants.")
            print("You may need to restart your Python session for the font changes to take effect.")
            return True
        else:
            print("No font variants were successfully installed.")
            return False
        
    except Exception as e:
        print(f"Error installing Roboto font: {e}")
        return False
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    """Main function to install the Roboto font"""
    if is_font_installed():
        print("Roboto font is already installed and available to matplotlib.")
        return
    
    success = install_roboto_font()
    
    if success:
        print("\nFont installation complete. You may need to restart your Python session or kernel for changes to take effect.")
    else:
        print("\nFont installation failed. You can try manually installing the Roboto font on your system.")
        print("Download from: https://fonts.google.com/specimen/Roboto")

if __name__ == "__main__":
    main() 