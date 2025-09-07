#!/usr/bin/env python3
"""
Script to fix NLTK download issues
"""
import os
import ssl

import nltk


def fix_nltk():
    print("Fixing NLTK download issues...")

    # Fix SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Create nltk_data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)

    # Download required data
    try:
        nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=False)
        print("✓ Successfully downloaded vader_lexicon")
    except Exception as e:
        print(f"✗ Failed to download vader_lexicon: {e}")

        # Manual download instructions
        print("\nManual download instructions:")
        print("1. Visit: https://www.nltk.org/nltk_data/")
        print("2. Search for 'vader_lexicon'")
        print("3. Download the zip file")
        print("4. Extract to: ~/nltk_data/sentiment/")


if __name__ == "__main__":
    fix_nltk()