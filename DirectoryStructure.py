import os
import argparse

def generate_tree(directory, prefix="", ignore_list=None):
    """
    A recursive function that generates a visual tree structure for a given directory.

    This function traverses the specified directory and prints its contents in a
    tree-like format, ignoring specified files and directories.

    Args:
        directory (str): The path to the directory to be traversed.
        prefix (str): The prefix string for the current level of the tree,
                      used for indentation and tree branch characters.
        ignore_list (list, optional): A list of directory/file names to ignore.
                                      Defaults to a common list for Python projects.
    """
    # Define a default list of common files/directories to ignore
    if ignore_list is None:
        ignore_list = [
            '__pycache__', '.pytest_cache', 'venv', '.venv',
            'env', '.env', '.git', '.vscode', '.idea'
        ]

    # Ensure the directory exists before proceeding.
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Get a list of all items (files and directories) in the current directory.
    try:
        # Filter out the items that are in the ignore list.
        items = [item for item in os.listdir(directory) if item not in ignore_list]
    except PermissionError:
        # If we don't have permission to read the directory, print an error and return.
        print(prefix + "└── [Permission Denied]")
        return

    # Sort items for a consistent, alphabetical order.
    items.sort()

    # Create the pointers for the tree structure.
    # '├── ' is used for all items except the last one in a directory list.
    # '└── ' is used for the very last item in the list.
    pointers = ['├── '] * (len(items) - 1) + ['└── ']

    for pointer, item in zip(pointers, items):
        # Construct the full path for the current item.
        path = os.path.join(directory, item)

        # Print the item with its corresponding tree pointer and prefix.
        print(f"{prefix}{pointer}{item}")

        # If the item is a directory, recursively call this function for that directory.
        if os.path.isdir(path):
            # The prefix for the next level depends on whether the current item is the last one.
            # If it is not the last ('├── '), we need a vertical line ('│   ') to show
            # that the parent branch continues.
            # If it is the last ('└── '), we use empty space ('    ') because the branch ends.
            extension = '│   ' if pointer == '├── ' else '    '
            generate_tree(path, prefix + extension, ignore_list)

def main():
    """
    Main function to parse command-line arguments and start the tree generation.
    """
    # Set up the argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(
        description="Generate a visual folder structure for a given path, ignoring common unimportant files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Examples:\n"
               "  # Generate structure for the current directory\n"
               "  python generate_structure.py\n\n"
               "  # Generate structure for a specific directory\n"
               "  python generate_structure.py /path/to/your/project"
    )

    # Add an argument for the directory path.
    # 'nargs'='?': Makes the argument optional.
    # 'default'='.' : If no path is provided, it defaults to the current directory ('.').
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='The root directory to generate the structure from. Defaults to the current directory.'
    )

    args = parser.parse_args()
    root_dir = args.path

    # Check if the provided path is a valid directory before starting.
    if not os.path.isdir(root_dir):
        print(f"Error: The path '{root_dir}' is not a valid directory.")
        return

    # Print the root directory name and start the recursive generation.
    # os.path.abspath ensures we have a clean, absolute path.
    # os.path.basename gets just the final directory name from the path.
    print(f"📁 {os.path.basename(os.path.abspath(root_dir))}")
    generate_tree(root_dir)

if __name__ == "__main__":
    main()