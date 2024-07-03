import streamlit as st
import os
import subprocess
from pathlib import Path
import git

# Helper function to clone repositories
def clone_repo(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        git.Repo.clone_from(repo_url, repo_dir)

# Define paths for repositories
WAV2LIP_REPO_URL = 'https://github.com/Rudrabha/Wav2Lip.git'
CODEFORMER_REPO_URL = 'https://github.com/sczhou/CodeFormer.git'

WAV2LIP_DIR = 'Wav2Lip'
CODEFORMER_DIR = 'CodeFormer'

# Clone repositories
clone_repo(WAV2LIP_REPO_URL, WAV2LIP_DIR)
clone_repo(CODEFORMER_REPO_URL, CODEFORMER_DIR)

# Define paths for models within the cloned repositories
WAV2LIP_MODEL_PATH = os.path.join(WAV2LIP_DIR, 'checkpoints/wav2lip.pth')
CODEFORMER_MODEL_PATH = os.path.join(CODEFORMER_DIR, 'checkpoints/codeformer.pth')

# Helper function to save uploaded files
def save_uploaded_file(uploaded_file, save_dir):
    save_path = Path(save_dir) / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Function to run the Wav2Lip model
def run_wav2lip(video_path, audio_path, output_path):
    command = [
        "python", os.path.join(WAV2LIP_DIR, "inference.py"), "--checkpoint_path", WAV2LIP_MODEL_PATH,
        "--face", str(video_path), "--audio", str(audio_path), "--outfile", str(output_path)
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running Wav2Lip model: {e.stderr}")

# Function to enhance video with CodeFormer
def enhance_video_with_codeformer(input_video_path, output_video_path):
    command = [
        "python", os.path.join(CODEFORMER_DIR, "inference_codeformer.py"), "--input_path", str(input_video_path),
        "--output_path", str(output_video_path), "--model_path", CODEFORMER_MODEL_PATH
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running CodeFormer model: {e.stderr}")

# Streamlit app
st.title("Wav2Lip and CodeFormer Integration")

st.write("Upload a video and an audio file to perform lip synchronization and enhance the video.")

video_file = st.file_uploader("Upload Video", type=["mp4"])
audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if video_file and audio_file:
    save_dir = "temp"
    os.makedirs(save_dir, exist_ok=True)

    video_path = save_uploaded_file(video_file, save_dir)
    audio_path = save_uploaded_file(audio_file, save_dir)
    output_path = Path(save_dir) / "final_output.mp4"

    st.write("Running Wav2Lip model...")
    run_wav2lip(video_path, audio_path, output_path)

    enhanced_output_path = Path(save_dir) / "enhanced_output.mp4"
    st.write("Enhancing video with CodeFormer model...")
    enhance_video_with_codeformer(output_path, enhanced_output_path)

    # Check if enhanced output file exists before displaying or downloading
    if enhanced_output_path.exists():
        st.success("Processing complete!")
        st.video(str(enhanced_output_path))

        with open(enhanced_output_path, "rb") as file:
            btn = st.download_button(
                label="Download Enhanced Video",
                data=file,
                file_name="enhanced_output.mp4",
                mime="video/mp4"
            )
    else:
        st.error("Error: Enhanced video file not found.")

