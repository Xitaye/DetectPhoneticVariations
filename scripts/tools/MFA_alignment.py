import os
import subprocess
import argparse
import shutil

# === MAIN LOGIC ===
def run_mfa_and_overall(base_path, dictionary="ipa_phone_mapping", acoustic_model="english_mfa"):
    temp_output = os.path.join(base_path, "__temp_mfa_output__")

    if not os.path.exists(base_path):
        print(f"❌ Error: The path '{base_path}' does not exist.")
        return

    # Make sure temp output directory exists
    os.makedirs(temp_output, exist_ok=True)

    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) and folder_name != "__temp_mfa_output__":
            print(f"\n🔄 Aligning: {folder_name}")
            folder_output = os.path.join(temp_output, folder_name)

            # Make sure this output folder is clean
            if os.path.exists(folder_output):
                shutil.rmtree(folder_output)
            os.makedirs(folder_output)

            # Run MFA
            cmd = [
                "mfa", "align", "--clean",
                folder_path,
                dictionary,
                acoustic_model,
                folder_output
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Finished alignment for: {folder_name}")

                # Copy all .TextGrid files to original folder
                for file in os.listdir(folder_output):
                    if file.endswith(".TextGrid"):
                        src = os.path.join(folder_output, file)
                        dst = os.path.join(folder_path, file)
                        shutil.copy(src, dst)
                        print(f"📄 Copied: {file} -> {folder_path}")

            except subprocess.CalledProcessError as e:
                print(f"❌ Error processing {folder_name}: {e}")

    # Optionally clean up
    shutil.rmtree(temp_output)
    print("\n🧹 Cleaned up temporary folder.")

def main():
    parser = argparse.ArgumentParser(description="Run MFA alignment over all folders in a base directory")
    parser.add_argument('--base_data_path', '-b', required=True,
                        help="Root path of Phase_1_recordings_clean")
    parser.add_argument('--dictionary', '-d', default="ipa_phone_mapping",
                        help="Name of the dictionary to use for alignment")
    parser.add_argument('--acoustic_model', '-a', default="english_mfa",
                        help="Name of the acoustic model")

    args = parser.parse_args()
    base_path = args.base_data_path
    dictionary = args.dictionary
    acoustic_model = args.acoustic_model

    run_mfa_and_overall(base_path, dictionary, acoustic_model)

    print("\n✅ All alignments completed successfully!")
    print("\nOutput files are located in the original folder structure.")

if __name__ == "__main__":
    main() 