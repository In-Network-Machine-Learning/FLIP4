########################################################################
# Copyright (c) Mingyuan Zang, Tomasz Koziak, Networks Technology and Service Platforms Group, Technical University of Denmark
# This work was partly done at Computing Infrastructure Group, University of Oxford
# All rights reserved.
# E-mail: mingyuanzang@outlook.com
# Licensed under the Apache License, Version 2.0 (the 'License')
########################################################################
import os
# Define the header content
# header="/*\n * Copyright (c) 2025 Your Company Name\n * All rights reserved.\n */\n\n"
header="""########################################################################
# Copyright (c) Mingyuan Zang, Tomasz Koziak, Networks Technology and Service Platforms Group, Technical University of Denmark
# This work was partly done at Computing Infrastructure Group, University of Oxford
# All rights reserved.
# E-mail: mingyuanzang@outlook.com
# Licensed under the Apache License, Version 2.0 (the 'License')
########################################################################\n"""
# Define file extensions to process
extensions = ['.py', '.sh', '.p4']


# Function to add header to a file
def add_header(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    if not content.startswith("/*"):
        with open(file_path, 'w') as file:
            file.write(header + content)
        print(f"Header added to {file_path}")

# Walk through the directory and add headers
for root, dirs, files in os.walk('.'):
    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            file_path = os.path.join(root, file)
            add_header(file_path)
