filepath_in = '2central_peaks.txt'
filepath_out = '2.txt'
with open(filepath_in, 'r') as file_in, open(filepath_out, 'w') as file_out:
    for line in file_in:
        data = line.split()  # splits the content removing spaces and final newline
        line_w_tabs = "\t".join(data) + '\n'
        file_out.write(line_w_tabs)
        
filepath_in = '3sidewings.txt'
filepath_out = '3.txt'
with open(filepath_in, 'r') as file_in, open(filepath_out, 'w') as file_out:
    for line in file_in:
        data = line.split()  # splits the content removing spaces and final newline
        line_w_tabs = "\t".join(data) + '\n'
        file_out.write(line_w_tabs)