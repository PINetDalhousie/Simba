import csv

# Open the input and output files
with open('../data/2023-11-28/fault-description.txt', 'r') as infile, open('../data/2023-11-28/fault-description.csv', 'w', newline='') as outfile:
    # Create a CSV writer
    writer = csv.writer(outfile)
    
    # Write the header row
    writer.writerow(['Fault Type', 'Start Time', 'End Time', 'Base Station'])
    
    # Process each line in the input file
    for line in infile:
        # Split the line into components
        components = line.strip().split()
        
        # Reformat the components into a CSV format
        fault_type = components[0]
        start_time = components[1].split('=')[1][:-2]  # Remove the 's' at the end
        end_time = components[2].split('=')[1][:-2]  # Remove the 's' at the end
        base_station = components[4].split('=')[1]  # Extract the integer value
        
        # Write the reformatted line to the CSV file
        writer.writerow([fault_type, start_time, end_time, base_station])