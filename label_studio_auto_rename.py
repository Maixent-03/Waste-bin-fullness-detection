"""
This script automates the renaming of exported Label Studio annotation files in the YOLO format.
It scans a specified directory for files matching the pattern:
    [hash]__ImagesV2%5CImage%20%28[number]%29.txt
and renames them to:
    Image ([number]).txt
This helps to clean up and organize files to make label file names match their corresponding image file names.
"""
import os
import re
import urllib.parse

def rename_files(directory_path):
    """
    Rename files that match the pattern: [hash]__ImagesV2%5CImage%20%28[number]%29.txt
    to: Image ([number]).txt
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist!")
        return
    
    # Pattern to match files like: [hash]__ImagesV2%5CImage%20%28[number]%29.txt
    pattern = r'^[a-f0-9]+__ImagesV2%5CImage%20%28(\d+)%29\.txt$'
    
    renamed_count = 0
    error_count = 0
    
    # Get all files in the directory
    files = os.listdir(directory_path)
    
    print(f"Found {len(files)} files in directory: {directory_path}")
    print("Processing files...")
    
    for filename in files:
        # Check if filename matches our pattern
        match = re.match(pattern, filename)
        
        if match:
            # Extract the number from the filename
            number = match.group(1)
            
            # Create new filename
            new_filename = f"Image ({number}).txt"
            
            # Full paths
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            try:
                # Check if new filename already exists
                if os.path.exists(new_path):
                    print(f"Warning: '{new_filename}' already exists. Skipping '{filename}'")
                    continue
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: '{filename}' → '{new_filename}'")
                renamed_count += 1
                
            except Exception as e:
                print(f"Error renaming '{filename}': {str(e)}")
                error_count += 1
        else:
            print(f"Skipping (doesn't match pattern): '{filename}'")
    
    print(f"\nRenaming complete!")
    print(f"Successfully renamed: {renamed_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"Files skipped (no pattern match): {len(files) - renamed_count - error_count} files")

def main():
    """Main function to run the file renaming script"""
    
    # Default directory - current directory + labels subfolder
    labels_directory = os.path.join(os.getcwd(), "labels")
    
    print("=" * 60)
    print("File Renaming Script")
    print("=" * 60)
    print(f"Target directory: {labels_directory}")
    print()
    
    # Ask user for confirmation
    user_input = input(f"Do you want to rename files in '{labels_directory}'? (y/n): ").lower().strip()
    
    if user_input == 'y' or user_input == 'yes':
        rename_files(labels_directory)
    else:
        print("Operation cancelled by user.")

if __name__ == "__main__":
    main()
