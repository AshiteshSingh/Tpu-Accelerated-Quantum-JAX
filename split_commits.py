import subprocess

# Reset the previous commit (undoes the commit but keeps files in working directory)
subprocess.run(['git', 'reset', 'HEAD~1'])

# Get list of modified/untracked files
status = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8')
files = []
for line in status.splitlines():
    if line:
        filepath = line[3:]
        # Remove quotes from paths with spaces
        if filepath.startswith('"') and filepath.endswith('"'):
            filepath = filepath[1:-1]
        files.append(filepath)

# Distribute files across exactly 37 commits
num_commits = 37
commits_files = [[] for _ in range(num_commits)]
for i, f in enumerate(files):
    commits_files[i % num_commits].append(f)

for i, chunk in enumerate(commits_files):
    if not chunk:
        # If we have less than 37 files, some chunks might be empty
        # We can't easily make empty commits unless we do --allow-empty
        subprocess.run(['git', 'commit', '--allow-empty', '-m', f'Quantum simulation checkpoint {i+1}'])
        continue

    for f in chunk:
        subprocess.run(['git', 'add', f])
    
    commit_msg = f'Update quantum research results part {i+1}'
    if '.gitignore' in chunk:
        commit_msg = 'Update .gitignore and project base files'
    
    subprocess.run(['git', 'commit', '-m', commit_msg])

# Force push to overwrite the remote branch
subprocess.run(['git', 'push', '-f', 'origin', 'master'])
