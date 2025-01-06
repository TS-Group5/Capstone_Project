def fetch_video(api_url, api_key, video_id, local_file_name):
    """
    Fetches video details from the API and downloads the video if the status is COMPLETED.

    Args:
        api_url (str): The base URL of the API.
        api_key (str): The API key for authorization.
        video_id (str): The ID parameter for the API request.
        local_file_name (str): The name of the file to save the downloaded video.
    """
    # Construct the API URL with the given ID
    url = f"{api_url}/{video_id}"
    headers = {
        "x-api-key": "sk-FcYNXOB3T22njoBg794L_w.KVnki7dwD4ZLf7yd6EB0G-rWoo_9SNeS"
    }

    # Step 1: Get the response from the API
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx
        data = response.json()

        # Step 2: Check if the status is "COMPLETED"
        if data.get("status") == "COMPLETED" and "outputUrl" in data:
            output_url = data["outputUrl"]

            # Step 3: Download the video from the output URL
            print("Status is COMPLETED. Downloading video...")
            video_response = requests.get(output_url, stream=True)
            if video_response.status_code == 200:
                # Step 4: Save the video to a local file
                with open(local_file_name, "wb") as video_file:
                    for chunk in video_response.iter_content(chunk_size=8192):
                        video_file.write(chunk)
                print(f"Video successfully downloaded and saved as '{local_file_name}'")
            else:
                print(f"Failed to download the video. Status code: {video_response.status_code}")
        else:
            print("Video is not ready or outputUrl is missing.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
