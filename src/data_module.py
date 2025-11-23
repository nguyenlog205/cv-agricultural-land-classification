

def read_and_process_data(file_path):
    """
    Reads data from the given file path and processes it.

    Args:
        file_path (str): The path to the data file.
    Returns:
        processed_data: The processed data.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Example processing: convert to uppercase
    processed_data = data.upper()
    
    return processed_data