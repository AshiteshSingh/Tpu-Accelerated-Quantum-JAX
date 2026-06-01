import re

with open('research_paper.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace local image paths with GitHub raw URLs
text = re.sub(r'\]\(tpu/plots/([^)]+)\)', r'](https://raw.githubusercontent.com/AshiteshSingh/Tpu-Accelerated-Quantum-JAX/main/tpu/plots/\1)', text)

# Remove the title and metadata block at the top since Hashnode handles that natively
# We'll strip everything before "## Abstract"
abstract_idx = text.find('## Abstract')
if abstract_idx != -1:
    text = text[abstract_idx:]

with open('hashnode_ready_blog.md', 'w', encoding='utf-8') as f:
    f.write(text)
print("Created hashnode_ready_blog.md")
