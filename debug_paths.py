import os

# We start looking from the current folder
current_dir = os.getcwd()
print(f"Current Working Directory: {current_dir}")

print("\n--- SEARCHING FOR 'with_mask' FOLDER ---")
found = False
# Walk through all folders to find where 'with_mask' is hiding
for root, dirs, files in os.walk(current_dir):
    if "with_mask" in dirs:
        print(f"\n✅ FOUND IT HERE: {os.path.join(root, 'with_mask')}")
        found = True
        # Stop after finding it to keep output clean
        break

if not found:
    print("\n❌ Could not find 'with_mask' folder anywhere inside this project.")
    print("Please verify you extracted the zip file into 'mask_detection_project'.")
