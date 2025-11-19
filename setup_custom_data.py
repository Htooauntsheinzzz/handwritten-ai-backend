"""
Setup Custom Data Directory Structure
Author: Htoo Aunt
Description: Creates folder structure for collecting custom handwriting samples
"""

import os

def setup_custom_data_folders():
    """Create the folder structure for custom digit data collection."""

    base_dir = 'custom_data'

    # Create train and test directories for each digit (0-9)
    for split in ['train', 'test']:
        for digit in range(10):
            path = os.path.join(base_dir, split, str(digit))
            os.makedirs(path, exist_ok=True)

    print("=" * 60)
    print("CUSTOM DATA FOLDER STRUCTURE CREATED")
    print("=" * 60)
    print("\nFolder structure:")
    print(f"""
{base_dir}/
├── train/
│   ├── 0/    <- Put training images of digit 0 here
│   ├── 1/    <- Put training images of digit 1 here
│   ├── 2/    <- Put training images of digit 2 here
│   ├── 3/    <- Put training images of digit 3 here
│   ├── 4/    <- Put training images of digit 4 here
│   ├── 5/    <- Put training images of digit 5 here
│   ├── 6/    <- Put training images of digit 6 here
│   ├── 7/    <- Put training images of digit 7 here
│   ├── 8/    <- Put training images of digit 8 here
│   └── 9/    <- Put training images of digit 9 here
└── test/
    ├── 0/    <- Put test images of digit 0 here
    ├── 1/    <- Put test images of digit 1 here
    ...
    └── 9/    <- Put test images of digit 9 here
""")

    print("\n" + "=" * 60)
    print("HOW TO COLLECT DATA")
    print("=" * 60)
    print("""
1. IMAGE REQUIREMENTS:
   - Format: PNG, JPG, or JPEG
   - Size: Any (will be resized to 28x28)
   - Color: Any (will be converted to grayscale)
   - Background: White background with black digit (preferred)

2. RECOMMENDED SAMPLES:
   - Training: 50-100+ images per digit
   - Testing: 10-20 images per digit
   - Total: ~600-1200 training + ~100-200 test images

3. TIPS FOR BETTER RESULTS:
   - Write digits clearly but naturally
   - Vary the writing style slightly
   - Center the digit in the image
   - Use consistent stroke thickness
   - Avoid cutting off parts of digits

4. NAMING CONVENTION:
   - Any name works (e.g., 0_001.png, digit_0_a.png)
   - Files are organized by folder, not filename

5. AFTER COLLECTING DATA:
   Run: python train_custom_model.py
""")
    print("=" * 60)

if __name__ == "__main__":
    setup_custom_data_folders()
