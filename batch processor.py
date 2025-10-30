import os
import shutil
import glob
from pathlib import Path

def split_images_into_batches(source_folder, destination_folder, num_batches=20):
    """
    Split image files from source folder into specified number of batch folders
    
    Args:
        source_folder (str): Path to folder containing images
        destination_folder (str): Path where batch folders will be created
        num_batches (int): Number of batch folders to create (default: 10)
    """
    
    # Common image file extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    
    # Get all image files from source folder
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_folder, extension)))
        image_files.extend(glob.glob(os.path.join(source_folder, extension.upper())))
    
    if not image_files:
        print(f"No image files found in {source_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    # Create batch folders
    batch_folders = []
    for i in range(num_batches):
        batch_folder = os.path.join(destination_folder, f"batch_{i+1}")
        Path(batch_folder).mkdir(parents=True, exist_ok=True)
        batch_folders.append(batch_folder)
        print(f"Created folder: {batch_folder}")
    
    # Calculate exact distribution using a more precise method
    total_files = len(image_files)
    base_size = total_files // num_batches
    remainder = total_files % num_batches
    
    # Create a list that specifies exactly how many files each batch should get
    batch_sizes = []
    for i in range(num_batches):
        if i < remainder:
            batch_sizes.append(base_size + 1)
        else:
            batch_sizes.append(base_size)
    
    print(f"\nDistributing {total_files} images into {num_batches} batches:")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total check: {sum(batch_sizes)} files")
    
    # Verify our math is correct
    assert sum(batch_sizes) == total_files, f"Math error: {sum(batch_sizes)} != {total_files}"
    
    # Split and copy files using exact indices
    current_index = 0
    
    for batch_num in range(num_batches):
        batch_size = batch_sizes[batch_num]
        batch_folder = batch_folders[batch_num]
        
        # Get exactly the right slice of files for this batch
        start_index = current_index
        end_index = current_index + batch_size
        batch_files = image_files[start_index:end_index]
        
        print(f"\nBatch {batch_num + 1}:")
        print(f"  Files {start_index + 1} to {end_index} ({len(batch_files)} images)")
        
        # Copy files to batch folder
        copied_count = 0
        for file_path in batch_files:
            filename = os.path.basename(file_path)
            destination_path = os.path.join(batch_folder, filename)
            
            try:
                shutil.copy2(file_path, destination_path)
                copied_count += 1
            except Exception as e:
                print(f"    Error copying {filename}: {e}")
        
        print(f"  Successfully copied: {copied_count} images")
        
        # Verify we copied the right amount
        if copied_count != batch_size:
            print(f"  WARNING: Expected {batch_size}, but copied {copied_count}")
        
        current_index += batch_size
    
    # Final verification
    print(f"\nSplit complete!")
    print(f"Processed {current_index} out of {total_files} files")
    
    # Count files in each batch folder to verify
    print("\nVerification - Files in each batch:")
    for i, batch_folder in enumerate(batch_folders):
        files_in_batch = len([f for f in os.listdir(batch_folder) 
                            if any(f.lower().endswith(ext[1:]) for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'])])
        expected = batch_sizes[i]
        status = "✓" if files_in_batch == expected else "✗"
        print(f"  batch_{i+1}: {files_in_batch} files (expected: {expected}) {status}")

# Example usage
if __name__ == "__main__":
    # Modify these paths according to your setup
    source_folder = "Dataset FA-1,FA-2 Computer vision based PPE detection Syystem\Train\images"          # Folder containing your 726 images
    destination_folder = "images_training"   # Where batch folders will be created
    
    # Split into 10 batches
    split_images_into_batches(source_folder, destination_folder, 20)

#to run label studio
# label-studio