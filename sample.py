import os

def print_tree(directory, prefix="", exclude_dirs={"venv", "__pycache__", "data","logs",".dvc",".gitignore"}):
    entries = sorted([e for e in os.listdir(directory) if e not in exclude_dirs])
    dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(directory, e))]
    
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1) and len(files) == 0
        print(f"{prefix}{'└── ' if is_last else '├── '}{d}/")
        print_tree(os.path.join(directory, d), prefix + ("    " if is_last else "│   "), exclude_dirs)
    
    for i, f in enumerate(files):
        is_last = i == len(files) - 1
        print(f"{prefix}{'└── ' if is_last else '├── '}{f}")

print_tree(".")
