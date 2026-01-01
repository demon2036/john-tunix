#!/usr/bin/env python3
"""Fix tunix sharding_utils.py for multi-host TPU support."""

SHARDING_FILE = "/home/john/tunix/tunix/sft/sharding_utils.py"

def main():
    print(f"Fixing {SHARDING_FILE} for multi-host TPU...")

    with open(SHARDING_FILE, 'r') as f:
        content = f.read()

    # Check if already fixed
    if 'addressable_data' in content:
        print("Already patched!")
        return

    # Find the shard_input function and add multi-host handling
    # We need to convert global arrays to local data before re-sharding

    old_code = '''  with jax.transfer_guard("allow"):
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        ),
        input_data,
    )'''

    new_code = '''  # Multi-host fix: convert global arrays to local data first
  def _to_local_data(x):
    if isinstance(x, jax.Array) and hasattr(x, 'sharding'):
      # Check if this is a global array spanning multiple processes
      if hasattr(x.sharding, 'mesh') and not x.sharding.mesh.empty:
        try:
          # Try to get addressable data for this process
          local_shards = x.addressable_shards
          if local_shards:
            # Concatenate all local shards
            local_data = np.concatenate([s.data for s in local_shards], axis=0)
            return local_data
        except Exception:
          pass
      # Fallback: convert to numpy
      return np.asarray(x)
    elif isinstance(x, np.ndarray):
      return x
    else:
      return x

  # Convert to local data first
  local_input_data = jax.tree.map(_to_local_data, input_data)

  with jax.transfer_guard("allow"):
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        ),
        local_input_data,
    )'''

    if old_code in content:
        content = content.replace(old_code, new_code)
    else:
        print("Could not find the exact code to replace. Trying alternative pattern...")
        # Try a more flexible replacement
        old_pattern = 'with jax.transfer_guard("allow"):\n    return jax.tree.map(\n        lambda x: jax.make_array_from_process_local_data('
        if old_pattern in content:
            # Find and replace the whole block
            start = content.find('with jax.transfer_guard("allow"):')
            if start != -1:
                # Find the end of the return statement (matching parentheses)
                end = content.find('\n\n', start)
                if end == -1:
                    end = content.find('\ndef ', start)
                if end != -1:
                    content = content[:start] + new_code + content[end:]

    with open(SHARDING_FILE, 'w') as f:
        f.write(content)

    print("Patch applied successfully!")

    # Verify
    with open(SHARDING_FILE, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if 'addressable_shards' in line or '_to_local_data' in line or 'local_input_data' in line:
                print(f"  Line {i}: {line.rstrip()}")

if __name__ == '__main__':
    main()
