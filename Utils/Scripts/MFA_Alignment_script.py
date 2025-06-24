import os
import sys
import subprocess
import shutil

def run_mfa_and_merge(base_path):
    dictionary = "ipa_phone_mapping"
    acoustic_model = "english_mfa"
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 /home/itayefrat/Util/MFA_Alignment_script.py /path/to/base_directory")        #Path to the data: /mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings/
    else:
        run_mfa_and_merge(sys.argv[1])