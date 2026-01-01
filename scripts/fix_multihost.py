#!/usr/bin/env python3
"""Fix tunix sampler.py for multi-host TPU support."""

import re
import sys

SAMPLER_FILE = "/home/john/tunix/tunix/generate/sampler.py"

def main():
    print(f"Fixing {SAMPLER_FILE} for multi-host TPU...")

    with open(SAMPLER_FILE, 'r') as f:
        content = f.read()

    # Check if already fixed with proper numpy conversion
    if 'np.asarray(multihost_utils' in content:
        print("Already patched with numpy conversion!")
        return

    # If previous patch exists, we need to update it
    if 'multihost_utils' in content:
        print("Updating existing patch to use numpy conversion...")
        # Replace previous process_allgather calls with numpy-converted versions
        content = content.replace(
            'out_tokens = multihost_utils.process_allgather(out_tokens, tiled=True)',
            '# Multi-host fix: gather and convert to local numpy array\n      out_tokens = np.asarray(multihost_utils.process_allgather(out_tokens, tiled=True))'
        )
        content = content.replace(
            'out_tokens = multihost_utils.process_allgather(out_tokens)',
            '# Multi-host fix: gather and convert to local numpy array\n      out_tokens = np.asarray(multihost_utils.process_allgather(out_tokens, tiled=True))'
        )
        content = content.replace(
            'lengths = multihost_utils.process_allgather(lengths, tiled=True)',
            'lengths = np.asarray(multihost_utils.process_allgather(lengths, tiled=True))'
        )
        content = content.replace(
            'lengths = multihost_utils.process_allgather(lengths)',
            'lengths = np.asarray(multihost_utils.process_allgather(lengths, tiled=True))'
        )
        content = content.replace(
            'gathered_tokens = multihost_utils.process_allgather(token_buffer, tiled=True)',
            'gathered_tokens = np.asarray(multihost_utils.process_allgather(token_buffer, tiled=True))'
        )
        content = content.replace(
            'gathered_tokens = multihost_utils.process_allgather(token_buffer)',
            'gathered_tokens = np.asarray(multihost_utils.process_allgather(token_buffer, tiled=True))'
        )
    else:
        # Fresh patch
        # Add import after 'import jax'
        content = content.replace(
            'import jax\n',
            'import jax\nfrom jax.experimental import multihost_utils\n'
        )

        # Fix line 768: out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
        old_line1 = 'out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)'
        new_line1 = '''# Multi-host fix: gather and convert to local numpy array
      out_tokens = np.asarray(multihost_utils.process_allgather(out_tokens, tiled=True))
      lengths = np.asarray(multihost_utils.process_allgather(lengths, tiled=True))'''
        content = content.replace(old_line1, new_line1)

        # Fix line 789: out_tokens.append(jax.device_get(token_buffer[start_idx:end_idx]))
        old_line2 = 'out_tokens.append(jax.device_get(token_buffer[start_idx:end_idx]))'
        new_line2 = '''# Multi-host fix: gather token_buffer first, then slice
        gathered_tokens = np.asarray(multihost_utils.process_allgather(token_buffer, tiled=True))
        out_tokens.append(gathered_tokens[start_idx:end_idx])'''
        content = content.replace(old_line2, new_line2)

    with open(SAMPLER_FILE, 'w') as f:
        f.write(content)

    print("Patch applied successfully!")

    # Verify
    with open(SAMPLER_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if 'process_allgather' in line or 'multihost_utils' in line:
                print(f"  Line {i}: {line.rstrip()}")

if __name__ == '__main__':
    main()
