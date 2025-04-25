import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time

# Set up client
client = OpenAI(
    api_key="sk-7defcd40a1dd469aafa128b1a0111788",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Load your data
df = pd.read_parquet("test-00000-of-00001.parquet")  # Replace with your filename
tqdm.pandas()

# Base video URL prefix
BASE_URL = "https://aisg-benchmark.s3.ap-southeast-2.amazonaws.com/Benchmark-AllVideos-HQ-Encoded-challenge/"

# Function to query model for one row
def query_model(row):
    try:
        video_url = BASE_URL + row["video_id"] + ".mp4"
        question = row["question"]
        prompt_instruction = row["question_prompt"]

        full_question = f"{question}\n{prompt_instruction}"

        completion = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a video question-answering assistant. Your goal is to watch short video clips and accurately answer user questions. The questions may involve observing visual events, identifying objects, comparing actions, counting repetitions, or reasoning about cause and effect. Always answer clearly and concisely based on what is visually observable in the video."
}]},
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": full_question}
                ]}
            ]
        )
        time.sleep(1.1)  # ~1 second per query for 60 QPM safety

        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"⚠️ Error with video_id: {row['video_id']}, question: {row['question']}")
        print("⚠️ Exception:", e)
        return "ERROR"

# Apply with progress bar
df["model_response"] = df.progress_apply(query_model, axis=1)

# Save output
df.to_csv("video_question_responses.csv", index=False)
print("✅ Saved model responses to video_question_responses.csv")
