from handler import handler
import json

# Input format
job = {
    "input": {
        "audio": "https://www.dropbox.com/scl/fi/cuzv8cct9vexljqo6r814/rooh.mp3?rlkey=gmknopc3kx05r45d7v8njssu6&st=jeerbylp&dl=1",  # or local path
        "text": r"https://www.dropbox.com/scl/fi/byhxis7loxegdceavoecv/rooh.txt?rlkey=l3k0ym08rgvk5rpnrjzt8p4t7&st=jdamiprf&dl=1",  # or direct text
        "language": "ara",
        "batch_size": 16,
        "output_format": "json",  # "srt", "json", or "both"
        "segment_by_lines": True
    }
}
output = handler(job)
json_data = output.get("json")
with open("output.json", "w" , encoding= "utf-8") as file:
    json.dump(json_data, file)
    file.close()