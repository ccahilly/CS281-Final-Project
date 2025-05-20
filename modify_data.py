import sys
import os

def modify_data(file_path):
    # Use a completely different output location - your desktop for example
    output_file = os.path.expanduser("~/Desktop/data_modified.txt")
    
    try:
        # Open the input file for reading and output file for writing
        with open(file_path, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Extract filename (first part before space)
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    # If no space, just write the line as is
                    outfile.write(line + '\n')
                    continue
                    
                filename = parts[0]
                value_part = parts[1]
                
                # Find the first comma in the value part
                comma_pos = value_part.find(',')
                if comma_pos != -1:
                    # Keep only what's before the comma
                    value_part = value_part[:comma_pos]
                
                # Write the modified line
                outfile.write(f"{filename} {value_part}\n")
                    
        print(f"Modified content saved to '{output_file}'")
        
        # Let's also verify by reading back the first few lines
        print("\nFirst few lines of the modified file:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # Print first 5 lines
                    print(line.strip())
                else:
                    break
                    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python modify_data.py file_path")
    else:
        file_path = sys.argv[1]
        modify_data(file_path)