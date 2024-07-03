import streamlit as st
import requests

# Function to submit POST request
def submit_request(audio_url, video_url):
    url = "https://api.synclabs.so/lipsync"
    payload = {
        "audioUrl": audio_url,
        "videoUrl": video_url,
        "model": "sync-1.6.0",
        "synergize": True
    }
    headers = {
        "x-api-key": "1ac2fdb8-ee69-42fc-91bc-f4793cf4138a",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code == 201:
        response_json = response.json()
        task_id = response_json.get('id')
        st.success(f"Task submitted successfully. Task ID: {task_id}")
        return task_id
    else:
        st.error(f"Failed to submit request. Status code: {response.status_code}")
        st.text(response.text)
        return None

# Function to fetch video URL using GET request
def fetch_video_url(task_id):
    url = f"https://api.synclabs.so/lipsync/{task_id}"
    headers = {"x-api-key": "1ac2fdb8-ee69-42fc-91bc-f4793cf4138a"}

    response = requests.request("GET", url, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        video_url = response_json.get('url')
        st.success("Video processing complete.")
        st.text(f"Video URL: {video_url}")
    else:
        st.error(f"Failed to fetch video URL. Status code: {response.status_code}")
        st.text(response.text)

# Streamlit app interface
def main():
    st.title("Lipsync Task Management")

    st.header("Submit Lipsync Task")
    audio_url = st.text_input("Enter Audio URL (MP3)")
    video_url = st.text_input("Enter Video URL (MP4)")

    if st.button("Submit Task"):
        if audio_url.strip() == "" or video_url.strip() == "":
            st.warning("Please enter both audio and video URLs.")
        else:
            st.info(f"Audio URL: {audio_url}")
            st.info(f"Video URL: {video_url}")
            task_id = submit_request(audio_url, video_url)

    st.header("Check Task Status and Get Video URL")
    st.info("Enter the task ID obtained after submitting the lipsync task.")
    task_id_input = st.text_input("Task ID")

    if st.button("Fetch Video URL"):
        if task_id_input.strip() == "":
            st.warning("Please enter a task ID.")
        else:
            fetch_video_url(task_id_input)

if __name__ == "__main__":
    main()
