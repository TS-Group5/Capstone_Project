import requests
import json
from src.db.db_connector  import getKey


def generate_formated_output_gemini(summary):
    # API endpoint
    api_key = getKey("gemini_api_key")
    url = (
        getKey("gemini_url")+r"{api_key}"
    )

    # Create the prompt for MCQ generation
    prompt_text = (
        f"You are a expert json formater, and you are generating json from the given text. {summary} "
        "Output format in JSON: example {Introduction:{ Caption: "", Audio:"" , Visual:""}, Experience: {Caption: "", Audio:"" , Visual:""}, Skills: {Caption: "", Audio:"" , Visual:""}, Achievement :{ Caption: "", Audio:"" , Visual:""},Goals :{ Caption: "", Audio:"" , Visual:""},Contact :{ Caption: "", Audio:"" , Visual:"" }}"
    )

    # Payload
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    # Set headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))
  
    if response.status_code == 200:
        data = response.json() 
        # Parse the response
        json_string = data['candidates'][0]['content']['parts'][0]['text']

# Remove the enclosing Markdown-like triple backticks and parse the JSON
        json_string = json_string.strip('```json\n').strip('```')
        output = json.loads(json_string)
        return output
    else:
        # Handle errors
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Example usage
  # Replace with your Gemini API key


    
    
      
            
      
